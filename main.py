import argparse
import gc
import math
import os
import shutil
import time
import datetime

import PIL.Image
######################

import torch
import torch.nn.functional as F  # 导入PyTorch中的函数式接口模块（torch.nn.functional），其中包含了一些常用的非线性激活函数、损失函数等。
import torch.backends.cudnn as cudnn  # 导入PyTorch的CUDNN后端模块（torch.backends.cudnn），用于提供针对CUDA加速的神经网络操作。NVIDIA CUDA®深度神经网络库（cuDNN）是GPU加速的用于深度神经网络的原语库。
import torch.autograd as autograd  # 导入PyTorch的自动求导模块（torch.autograd），它提供了计算图和自动微分的功能，用于求解梯度和进行反向传播。
import torchvision.datasets
import torchvision.transforms as transforms  # 导入torchvision库中的数据预处理模块（torchvision.transforms），用于对图像数据进行常见的预处理操作，如裁剪、缩放、翻转等。

######################

import numpy as np
from loss_functions import *
from models import SFMnet as SFMnet
import flow_transforms

# SummaryWriter类用于创建一个TensorBoard的事件文件写入器，它可以将训练过程中的损失、准确率、模型参数等信息写入到事件文件中。
# tensorboardX库是对TensorBoard的一个轻量级封装，它提供了一个用于记录和可视化PyTorch模型训练过程的接口。
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter  # 使用的pytorch 1.8.0 （still need install tensorboard）

from utils import tensor2array
import imageio as io
import logging  # 输出运行日志
from lib.config import cfg, cfg_from_file, save_config_to_file  # 加载cfg
from demon_metrics import compute_motion_errors, l1_inverse, scale_invariant, abs_relative
import random
from models.SFMnet import Pose2RT
from flow_viz import flow_to_image

from KITTI_loader import KITTIVOLoaderGT, KITTIRAWLoaderGT

# from DEMON_loader import DEMON_GT_LOADER
# from flow_training import train_flow

###################### for mixed precision training

# torch.cuda.amp.GradScaler是用于自动混合精度训练的工具类。
# 在这段代码中，首先尝试导入torch.cuda.amp.GradScaler，如果导入失败（即在PyTorch版本低于1.6时），
# 则定义了一个名为GradScaler的虚拟类，使其具有与torch.cuda.amp.GradScaler相同的方法和属性。
# 这样，在代码其他部分使用GradScaler时，即使没有torch.cuda.amp.GradScaler也不会报错。
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# 在PyTorch 1.6及更高版本中，torch.cuda.amp.autocast是用于自动混合精度训练的上下文管理器。
# 在这段代码中，首先尝试导入torch.cuda.amp.autocast，如果导入失败（即在PyTorch版本低于1.6时），
# 则定义了一个名为autocast的虚拟类，使其具有与torch.cuda.amp.autocast相同的上下文管理功能。
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

parser = argparse.ArgumentParser(description='Structure from Motion network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建了一个命令行参数解析器
parser.add_argument('--data', dest='data', metavar='DIR', help='path to train dataset')  # 用于制定训练数据文件的路径
parser.add_argument('--cfg', dest='cfg', default=None, type=str)  # 用于指定配置文件的路径。

###################### 预训练模型选择（sfm模型，光流模型，距离深度模型，保存验证图像）
parser.add_argument('--pretrained', dest='pretrained', default=None, metavar='PATH',
                    help='path to pre-trained SFMnet model')  # 用于指定预训练的SFMnet模型文件路径。（sfm模型）
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained epiflow model')  # 用于指定预训练的epiflow模型文件路径。（光流模型）
parser.add_argument('--pretrained-depth', dest='pretrained_depth', default=None, metavar='PATH',
                    help='path to pre-trained dspnet model')  # 用于指定预训练的dspnet模型文件路径。（距离深度模型）

parser.add_argument('--nlabel', dest='nlabel', type=int, default=64, help='number of label')  # 用于指定标签的数量，参数类型为整数，默认值为64。
parser.add_argument('--save-images', dest="save_images", default=False, action='store_true', help='save validation images')  # 设置“保存验证图像”属性为True。

###################### 深度模型参数设置（优化器，进程数量，训练轮数，每轮训练本数，小批量样本数）
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithms')  # 用于更新网络权重，选择优化求解器算法，可选值为adam和sgd，默认值为adam。
parser.add_argument('-j', '--workers', dest='workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')  # 用于指定数据加载的工作进程数量，默认值为8。
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')  # 用于指定总共运行的训练轮数，默认值为300
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')  # 用于指定手动设置的起始训练轮数，在重新开始训练时很有用，默认值为0。
parser.add_argument('--epoch-size', dest='epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')  # 用于手动设置每轮训练的样本数（如果设置为0，则与数据集的大小匹配）
parser.add_argument('-b', '--batch-size', dest='batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size')  # 用于指定每个小批量的样本数，默认值为4。
# SGD算法参数设置 （学习率，Adam alpha beta参数，权重衰减正则化因子）
parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')  # 用于指定初始学习率，默认值为0.001。
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')  # 用于SGD算法的动量参数，用于Adam算法的alpha参数，默认值为0.9。
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')  # 用于Adam算法的beta参数，默认值为0.999
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')  # 用于指定权重衰减，默认值为4e-4。优化算法中调整正则化因子λ的大小来控制权重衰减的影响程度。

######################  日志输出频率，是否验证评估模型，是否训练光流，是否训练深度
parser.add_argument('--print-freq', '-p', dest='print_freq', default=50, type=int,
                    metavar='N', help='logger.info frequency')  # 用于指定日志信息输出的频率，默认为每50个批次输出一次。
parser.add_argument('-v', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')  # 该参数是一个开关，如果提供了该命令行参数，则设置validate属性为True，用于在验证集上评估模型。
parser.add_argument('--div-flow', default=1,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')  # 用于指定将光流值除以的值。原始值为20，但使用批归一化后，可以将其设置为1以获得更好的结果。
parser.add_argument('--milestones', default=[2, 5, 8], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')  # 用于指定在哪些训练轮数时将学习率除以2。该参数接受一个或多个整数值，并将它们存储在列表中。
parser.add_argument('--fix-flownet', dest='fix_flownet', action='store_true', help='do not train flownet')  # 该参数是一个开关，如果提供了该命令行参数，则设置fix_flownet属性为True，不训练光流网络。
parser.add_argument('--fix-depthnet', dest='fix_depthnet', action='store_true', help='do not train depthnet')  # 该参数是一个开关，如果提供了该命令行参数，则设置fix_depthnet属性为True，不训练深度网络。

best_EPE = -1  # 用于追踪最佳的端点误差(EPE, Endpoint Error)，初始值设置为-1。在训练过程中，当得到更低的EPE时，可以更新该变量的值。
n_iter = 0  # 用于记录迭代次数的计数器，初始值设置为0。在每次迭代或训练步骤完成后，可以通过增加n_iter来更新迭代次数。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测当前主机是否支持CUDA，并将设备类型设置为"CUDA"（即使用GPU）或"CPU"。
print("device:", device)
print("count:", torch.cuda.device_count())

"""
# 这段代码是一个用于训练和测试计算机视觉模型的主函数。以下是主要步骤的简要说明：
# 1 解析命令行参数，并根据配置文件设置日志文件和保存路径。
# 2 设置TensorBoard记录训练过程。
# 3 定义图像预处理和数据加载的方法。
# 4 根据配置文件选择数据集加载方式。
# 5 创建模型，并根据需要加载预训练模型。..
# 6 定义优化器和学习率调度器。
# 7 进行训练和验证循环，保存最佳模型。其中包含了训练光流或深度估计模型的逻辑判断，以及保存和加载模型的步骤。
"""


# python main.py -v -b 1 -p 1 --nlabel 128 --data /media/lzz/My_Passport/kitti/kitti_raw/ --cfg cfgs/kitti.yml --pretrained split_files/kitti.pth.tar

def main():
    global args, best_EPE, save_path, n_iter  # 命令行参数, 最佳的端点误差, 模型保存路径， 迭代次数
    args = parser.parse_args()  # 解析命令行参数

    # python main.py -b 32 --lr 0.0005 --nlabel 128 --fix-flownet --data /media/lokia/My_Passport/kitti/kitti_raw/ --cfg cfgs/kitti.yml --pretrained-depth split_files/depth_init.pth.tar --pretrained-flow split_files/flow_init.pth.tar

    args.batch_size = 32
    args.epoch_size = 0
    args.lr = 0.0005
    args.nlabel = 128  # 用于指定深度候选平面的数量，参数类型为整数，默认值为64。
    args.data = "/media/lokia/My_Passport/kitti/kitti_raw/"
    args.cfg = "cfgs/kitti.yml"
    args.pretrained_depth = "split_files/depth_init.pth.tar"
    args.pretrained_flow = "split_files/flow_init.pth.tar"
    args.fix_flownet = True
    args.print_freq = 1  # 指定日志信息输出的频率
    # args.pretrained = "" # SFMnet训练过的模型
    # args.save_images = True

    ### 1 解析命令行参数，并根据配置文件设置日志文件和保存路径。
    # 配置加载
    if args.cfg is not None:
        cfg_from_file(args.cfg)
        assert cfg.TAG == os.path.splitext(os.path.basename(args.cfg))[0], 'TAG name should be file name'

    # 创建输出目录
    save_path = os.path.join("output", cfg.TAG)  # output/kitti
    if not os.path.exists(save_path): os.makedirs(save_path)

    # set log files 
    log_file = os.path.join(save_path, 'log_train.txt')
    logger = create_logger(log_file)  # 设置logger参数
    logger.info('**********************Start logging**********************')
    logger.info('=> will save everything to {}'.format(save_path))

    # save configs for future reference
    for _, key in enumerate(args.__dict__):
        logger.info((key, args.__dict__[key]))
    save_config_to_file(cfg, logger=logger)

    scaler = GradScaler(enabled=cfg.MIXED_PREC)  # 使用GradScaler类创建一个名为scaler的对象，其enabled属性值为cfg.MIXED_PREC（配置中的混合精度） default False

    ### 2 设置TensorBoard记录训练过程。 这些对象将用于记录训练和测试期间的摘要信息
    train_writer = SummaryWriter(log_dir=os.path.join(save_path, 'train', datetime.datetime.now().strftime('%Y_%m_%d'), datetime.datetime.now().strftime('%H_%M_%S')))  # output/kitti/train
    test_writer = SummaryWriter(log_dir=os.path.join(save_path, 'test', datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))  # output/kitti/test

    ### 3 定义图像预处理和数据加载的方法。
    # 图像加载和归一化：
    # 定义了input_transform和depth_transform两个变换。

    # input_transform是一个由多个变换组成的流水线，用于将输入图像转换为张量并进行归一化。
    # 首先使用ArrayToTensorCo将图像数组转换为张量，然后使用NormalizeCo进行归一化操作。
    input_transform = flow_transforms.Compose([
        flow_transforms.ArrayToTensorCo(),
        flow_transforms.NormalizeCo(mean=[0, 0, 0], std=[255, 255, 255]),
        flow_transforms.NormalizeCo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # depth_transform只是将深度图像数组转换为张量。
    depth_transform = flow_transforms.Compose([flow_transforms.ArrayToTensorCo()])

    # 设置图像尺寸变换：
    # co_transform_train和co_transform_val分别定义了训练集和验证集图像的预处理变换。
    co_transform_train = flow_transforms.ComposeCo([
        flow_transforms.RandomCropCo((cfg.TRAIN_SIZE[0], cfg.TRAIN_SIZE[1])), ])  # 对于训练集，使用RandomCropCo随机裁剪图像到指定的大小 [256,512]
    co_transform_val = flow_transforms.ComposeCo([
        flow_transforms.CenterCropCo((cfg.VAL_SIZE[0], cfg.VAL_SIZE[1])), ])  # 对于验证集，使用CenterCropCo居中裁剪图像到指定的大小 [370,1224]

    # 打印日志：
    # 使用logger.info函数记录打印信息，表明正在从args.data指定的路径中获取图像对。
    logger.info("=> fetching img pairs in '{}'".format(args.data))

    #### data loader 4 根据配置文件选择数据集加载方式。
    #### 这段代码的主要目的是根据配置文件选择数据集加载方式，并创建训练集和验证集的数据加载器。同时，根据需要生成DEMON姿态或深度，并打印相关信息。

    # 这些对象使用之前设置的图像变换进行预处理，包括输入图像转换、深度图像转换和图像尺寸变换。
    if cfg.KITTI_RAW_DATASET:  # 使用 KITTI_RAW_DATASET
        train_set = KITTIRAWLoaderGT(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_train, train=True)
        # print(train_set[0])
        val_set = KITTIRAWLoaderGT(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_val, train=False)

    elif cfg.DEMON_DATASET:
        if not args.validate: train_set = DEMON_GT_LOADER(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_train, train=True, ttype='train.txt')
        val_set = DEMON_GT_LOADER(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_val, train=False, ttype='test.txt')
    else:
        train_set = KITTIVOLoaderGT(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_train, train=True)
        val_set = KITTIVOLoaderGT(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_val, train=False)
        # print(train_set)

    if cfg.GENERATE_DEMON_POSE_TO_SAVE or cfg.GENERATE_DEMON_POSE_OR_DEPTH:
        val_set = DEMON_GT_LOADER(root=args.data, transform=input_transform, target_transform=depth_transform, co_transform=co_transform_val, train=False, ttype='train.txt', return_path=True)

    # 使用logger.info函数打印数据集样本数量信息，包括总样本数、训练样本数和测试样本数
    try:
        logger.info('{} samples found, {} train samples and {} test samples '.format(len(val_set) + len(train_set), len(train_set), len(val_set)))
    except:
        logger.info('{} test samples '.format(len(val_set)))

    # 使用torch.utils.data.DataLoader创建训练集和验证集的数据加载器。可以设置批量大小、并行加载的工作线程数、是否将数据保存到固定内存位置、是否打乱数据等选项。
    # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    if not args.validate:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   num_workers=args.workers, pin_memory=False, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                             num_workers=args.workers, pin_memory=False, shuffle=False)

    ##################################################################

    ### 5 创建SFMnet模型
    model = SFMnet(args.nlabel)  # 使用SFMnet类创建一个新的模型对象，并传入参数args.nlabel作为平面候选的数量。

    if args.pretrained:  # 用于指定预训练的SFMnet模型文件路径。（sfm模型） None
        network_data = torch.load(args.pretrained)
        logger.info("=> using pre-trained model '{}'".format(args.pretrained))
        model.load_state_dict(network_data["state_dict"], strict=False)  # 在加载过程中允许出现部分权重参数匹配不成功的情况
    else:
        network_data = None
        logger.info("=> creating new model")

    ### 6 定义优化器。
    # optimizer
    assert (args.solver in ['adam', 'sgd'])  # adam
    logger.info('=> setting {} solver'.format(args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # 7. 多GPU并行
    model = torch.nn.DataParallel(model).cuda()  # 使用torch.nn.DataParallel对模型进行多GPU并行处理，该操作会将模型复制到所有可用的GPU设备上。

    # 8. 根据需要加载预训练模型
    # 该段代码的主要目的是根据命令行参数选择是否加载预训练的光流网络模型和深度估计网络模型，并将加载的权重参数应用到相应的网络组件中。
    if args.pretrained_flow:  # 加载预训练的光流网络模型
        temp_dict = {}
        pretrained_dict = torch.load(args.pretrained_flow)
        # TBD: remove these dummy codes
        flag = False
        for key in pretrained_dict['state_dict'].keys():
            if 'flow_estimator' in key:  # not have
                flag = True;
                temp_dict[key.replace('flow_estimator.', '')] = pretrained_dict['state_dict'][key]  # 预训练模型的参数字典替换为新的temp_dict
        if flag:  # 'flow_estimator' don't exist,so flag is still False
            pretrained_dict['state_dict'] = temp_dict  # 只加载与当前程序相匹配的参数，并避免因键名不匹配而导致的错误。

        model_dict = model.module.flow_estimator.state_dict()
        model_dict.update(pretrained_dict)
        model.module.flow_estimator.load_state_dict(model_dict['state_dict'], strict=False)
        logger.info("=> using pre-trained flow network: '{}'".format(args.pretrained_flow))

    if args.pretrained_depth:  # "split_files/depth_init.pth.tar
        pretrained_dict = torch.load(args.pretrained_depth)
        model_dict = model.module.depth_estimator.state_dict()
        model_dict.update(pretrained_dict)
        model.module.depth_estimator.load_state_dict(model_dict['state_dict'], strict=False)
        logger.info("=> using pre-trained dpsnet: '{}'".format(args.pretrained_depth))
    # 9 开启 cuDNN 的自动优化
    cudnn.benchmark = True  # 将 cudnn.benchmark 设置为 True，这将使得 cuDNN 根据输入数据的大小和形状自动选择最适合的卷积实现算法。
    # 10 创建学习率调度器
    # 使用 torch.optim.lr_scheduler.MultiStepLR 类创建一个多步学习率调度器对象 scheduler。
    # 该调度器会根据给定的里程碑（milestones）和衰减因子（gamma），在训练过程中自动调整优化器的学习率。当训练步数达到里程碑时，学习率会按照衰减因子进行缩放。
    # MILESTONES: [3,8]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=0.5)

    epoch = 0

    ### 11 validate 加载最佳模型
    if cfg.GENERATE_DEMON_POSE_TO_SAVE:  # False
        from generate_demon_pose import generate_DEMON_pose
        with torch.no_grad():
            best_EPE = generate_DEMON_pose(val_loader, model, epoch, logger)
        return
    # 根据配置参数cfg.SAVE_POSE的值，选择是否加载姿态信息。
    if args.validate:  # False
        with torch.no_grad():
            if cfg.SAVE_POSE:
                best_EPE = save_pose(val_loader, model, epoch, logger)
            else:
                best_EPE = validate(val_loader, model, epoch, logger)
        return

    ##################################################################
    ### 12 进入训练和验证循环，保存最佳模型。
    for epoch in range(args.start_epoch, args.epochs):
        # 设置光流网络和深度网络的参数是否需要固定（即不进行梯度更新）
        # 13 选择训练光流网络or深度网络
        # fix the weights
        if args.fix_flownet:  # True 固定光流网络
            for fparams in model.module.flow_estimator.parameters():  fparams.requires_grad = False
        if args.fix_depthnet:  # False 不固定深度网络
            for fparams in model.module.depth_estimator.parameters():  fparams.requires_grad = False

        is_best = False  # 用于记录当前是否是最佳模型

        # 14 训练网络，返回loss
        if cfg.TRAIN_FLOW:  # False 如果配置参数cfg.TRAIN_FLOW为True，则执行光流训练操作（固定光流网络）
            # train optical flow on Demon datasets
            train_loss, n_iter = train_flow(train_loader, model, optimizer, epoch,
                                            train_writer, scaler, logger=logger, n_iter=n_iter, args=args)  # 调用train_flow函数，在训练集train_loader上对模型model进行光流训练。该函数还会使用优化器optimizer、写入器train_writer、混合精度训练的缩放器scaler以及记录日志的logger进行训练，并返回训练损失和训练迭代次数。
        else:  # 如果cfg.TRAIN_FLOW为False，则执行深度训练操作（不固定深度网络）
            # train depth
            train_loss, scheduler = train_epoch(train_loader, model, optimizer, epoch,
                                                train_writer, scaler, logger, scheduler=scheduler)  # 调用train_epoch函数，在训练集train_loader上对模型model进行深度训练。该函数还会使用优化器optimizer、写入器train_writer、混合精度训练的缩放器scaler、记录日志的logger和学习率调度器scheduler进行训练，并返回训练损失和更新后的学习率调度器。

        # 15 调用scheduler.step()来更新学习率调度器的状态，以便在下一个迭代中使用新的学习率
        scheduler.step()

        # 16 调用save_checkpoint函数保存当前模型的checkpoint。
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.module.state_dict()},
                        is_best, filename='checkpoint{}.pth.tar'.format(epoch))  # 该函数会保存当前迭代的epoch数、模型的状态字典以及一个指示是否是最佳模型的布尔值is_best。保存的文件名为checkpoint{}.pth.tar，其中{}会被当前迭代的epoch数替换。

        # 17 使用验证集val_loader对模型model进行验证。
        with torch.no_grad():
            best_EPE = validate(val_loader, model, epoch, logger)  # 调用validate函数，并使用验证集验证模型的性能。该函数还会使用当前迭代的epoch数和记录日志的logger。返回的best_EPE为最佳验证误差（如平均点误差）。


def train_epoch(train_loader, model, optimizer, epoch, train_writer, scaler, logger=None, scheduler=None):
    global n_iter, args
    # 用于计算和存储平均值和当前值的辅助类AverageMeter
    batch_time = AverageMeter()  # 记录训练时间
    data_time = AverageMeter()  # 数据读取时间
    losses = AverageMeter()  # loss工具

    # epoch_size表示每个epoch需要遍历的batch数量，如果args.epoch_size为0，则遍历整个数据集
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
    # switch to train mode
    model.train()
    end = time.time()
    # 使用i来记录当前是第几个batch
    # input表示输入数据，intrinsics表示相机内参，poses表示相机位姿，pred_poses表示预测的相机位姿，depth_gt表示ground truth深度图。
    for i, (input, intrinsics, poses, pred_poses, depth_gt) in enumerate(train_loader):

        # print("train_epoch:", i)
        optimizer.zero_grad()
        data_time.update(time.time() - end)

        # two frames
        input0 = input[0].to(device);
        input1 = input[1].to(device)

        # print(i, img_path[0][0])
        # img = PIL.Image.open(img_path[0][0])
        # img.show()

        # add random noise
        # 对于每个输入图像，都添加一个随机噪声，以增强模型的鲁棒性
        stdv = np.random.uniform(0.0, 3.0 / 255)
        input0 = (input0 + stdv * torch.randn(*input0.shape).cuda()).clamp(-1, 1)
        input1 = (input1 + stdv * torch.randn(*input1.shape).cuda()).clamp(-1, 1)

        # forward and backward pose
        pose_gt_fw = poses[0].to(device)  # reference frame -> target frame
        pose_gt_bw = poses[1].to(device)  # target frame -> reference frame

        # we could save predicted poses as local files, to reduce training time 
        pred_pose_fw = pred_poses[0].to(device) if pred_poses is not None else None
        pred_pose_bw = pred_poses[1].to(device) if pred_poses is not None else None

        depth_fw_gt = depth_gt[0].to(device);  # target frame 视差图 # gt_depth1 = gt_depth2.copy()
        depth_bw_gt = depth_gt[1].to(device)  # reference frame # gt_depth2 is true

        raw_shape = input1.shape;
        height_raw = raw_shape[2];
        width_raw = raw_shape[3]

        # DICLFlow limits the size of input frames, so we pad the inputs
        # feel free to delete this part if using other flow estimator
        # 这段代码是为了将输入帧进行填充，以满足DICLFlow流估计器对输入帧大小的限制。如果您使用其他流估计器，可以删除这部分代码。
        height_new = int(np.ceil(raw_shape[2] / 128) * 128)
        width_new = int(np.ceil(raw_shape[3] / 128) * 128)
        padding = (0, width_new - raw_shape[3], 0, height_new - raw_shape[2])
        input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)
        input0 = torch.nn.functional.pad(input0, padding, "replicate", 0)

        # 需要注意的是，这里的"backward" 表示从输入帧1（参考帧）到输入帧0（目标帧），而不是指反向传播
        # input0是目标帧，input1是参考帧。
        # intrinsics是相机内参，用于将像素坐标转换为相机坐标。
        # pose_gt_bw是反向位姿，表示从目标帧到参考帧的相对变换。
        # pred_pose_bw是预测的反向位姿。
        ########################## model prediction #####################################
        flow_2D_bw, pose_bw, depth_bw, depth_bw_init, rot_and_trans = model(input1, input0, intrinsics, pose_gt_bw, pred_pose_bw, cfg.GT_POSE, h_side=raw_shape[2], w_side=raw_shape[3], logger=logger)
        # 光流，位姿，深度，深度初始化，旋转和平移

        # 这两行代码在SFMnet.forward()中运行
        # intrinsics_invs = intrinsics.inverse().float().to(device)
        # intrinsics = intrinsics.float().to(device)

        # filter out meaningless depth values (out of range, NaN or Inf)
        ###### 这段代码的作用是过滤掉深度图中不合法的值。######
        # 在这里，depth_bw_gt 和 depth_fw_gt 分别代表反向深度图和正向深度图，它们的值表示从目标帧到参考帧和从参考帧到目标帧的深度。
        # 深度图是许多计算机视觉任务的重要输入，如物体检测、语义分割、视觉里程计等。然而，在现实应用中，深度图可能会受到各种因素的干扰而产生不合法的值，如传感器噪声、运动模糊或遮挡等。
        # 为了避免这些不合法的值对模型训练造成影响，需要在训练过程中将其过滤掉。具体而言，这段代码使用了一系列条件判断，包括：
        # depth_bw_gt <= args.nlabel * cfg.MIN_DEPTH: 深度值不能超过最大深度。
        # depth_bw_gt >= cfg.MIN_DEPTH: 深度值不能小于最小深度=1
        # depth_bw_gt == depth_bw_gt: 深度值必须为非NaN值。
        # 这些条件判断将生成一个布尔类型的掩码数组，其中每个元素都表示深度图中对应位置的像素是否合法。对于不合法的像素，其在掩码数组中的值为 False，否则为 True。
        # 这个掩码数组可以用于后续的损失计算和优化过程，例如将不合法的像素对应的深度值设置为 0 或其他默认值，以避免其对损失函数的计算造成影响。
        mask_bw = (depth_bw_gt <= args.nlabel * cfg.MIN_DEPTH) & (depth_bw_gt >= cfg.MIN_DEPTH) & (depth_bw_gt == depth_bw_gt)
        mask_fw = (depth_fw_gt <= args.nlabel * cfg.MIN_DEPTH) & (depth_fw_gt >= cfg.MIN_DEPTH) & (depth_fw_gt == depth_fw_gt)

        if not args.fix_depthnet:
            ##################  Compute Depth Loss ########################
            with autocast(enabled=cfg.MIXED_PREC):  # 上下文管理器，用于开启或关闭混合精度训练（mixed-precision training）
                if cfg.RESCALE_DEPTH:  # True 根据配置文件中的设置 cfg.RESCALE_DEPTH 来决定是否对深度图进行重新缩放
                    # the translation scale of ground truth poses
                    # 首先，计算了目标帧到参考帧的位姿变换的尺度，并构建了一个尺度掩码，即 scale_mask。
                    # 然后，计算了尺度的归一化比例 rescale_ratio，并使用该比例将深度图 depth_bw 进行缩放。
                    scale = torch.norm(pose_gt_bw[:, :, -1:].squeeze(-1), dim=-1)
                    scale_mask = (scale > cfg.MIN_TRAIN_SCALE) & (scale < cfg.MAX_TRAIN_SCALE)
                    rescale_ratio = scale / cfg.NORM_TARGET

                    # to keep scale consistent
                    depth_bw = depth_bw * rescale_ratio.view(scale.shape[0], 1, 1, 1)

                    # 最后，根据 cfg.RESCALE_DEPTH_REMASK 的设置，
                    # 重新检查了缩放后深度图的边界，并生成了一个深度图的掩码，即 mask_bw。
                    if cfg.RESCALE_DEPTH_REMASK:  # False
                        # recheck the boundary of scales, unnecessary
                        depth_bw_gt_rescale = depth_bw_gt / (rescale_ratio.view(-1, 1, 1, 1))
                        mask_bw = (depth_bw_gt_rescale <= args.nlabel * cfg.MIN_DEPTH) & (depth_bw_gt_rescale >= cfg.MIN_DEPTH) & (depth_bw_gt_rescale == depth_bw_gt_rescale)
                        train_writer.add_scalar('max_gt_depth', depth_bw_gt_rescale.max().item(), n_iter)
                        train_writer.add_scalar('min_gt_depth', depth_bw_gt_rescale[depth_bw_gt_rescale > 0].min().item(), n_iter)
                        mask_bw = mask_bw.detach()

                    # pick the valid ones for optimization
                    # 使用尺度掩码 scale_mask 和深度图掩码 mask_bw 来选择有效的像素，用于优化
                    pred_init_toloss = depth_bw_init[scale_mask][mask_bw[scale_mask]]
                    pred_toloss = depth_bw[scale_mask][mask_bw[scale_mask]]
                    gt_toloss = depth_bw_gt[scale_mask][mask_bw[scale_mask]]
                else:  # False
                    # pick the valid ones for optimization
                    # 在没有对深度图进行重新缩放的情况下，直接计算了深度图中的合法像素，以及对应的预测深度 pred_toloss 和真实深度 gt_toloss。
                    scale = torch.norm(pose_gt_bw[:, :, -1:].squeeze(-1), dim=-1)
                    scale_mask = (scale > cfg.MIN_TRAIN_SCALE)
                    pred_init_toloss = depth_bw_init[scale_mask][mask_bw[scale_mask]]
                    pred_toloss = depth_bw[scale_mask][mask_bw[scale_mask]]
                    gt_toloss = depth_bw_gt[scale_mask][mask_bw[scale_mask]]

                # 接下来，根据 DPSNet 论文的设置，计算了深度损失。
                # loss_depth_init 是初始深度图和真实深度图之间的平滑 L1 损失。
                # loss_depth_out 是优化后的深度图和真实深度图之间的平滑 L1 损失。
                # 然后将两个损失加权相加得到总的深度损失 loss_depth。
                # follow the setting of DPSNet
                loss_depth_init = 0.7 * (F.smooth_l1_loss(pred_init_toloss, gt_toloss))
                loss_depth_out = F.smooth_l1_loss(pred_toloss, gt_toloss)
                loss_depth = loss_depth_out + loss_depth_init

                # 最后，将深度损失的相关信息写入训练日志中，同时将 loss 赋值为 loss_depth，以便后续使用。
                train_writer.add_scalar('depth_init', loss_depth_init.item(), n_iter)
                train_writer.add_scalar('depth_out', loss_depth_out.item(), n_iter)

            loss = loss_depth

        # 这段代码用于计算姿态损失（pose loss）。
        # 首先，检查变量 rot_and_trans 是否为非空。如果不为空，则执行以下代码块。
        # 代码开始通过调用函数 Pose2RT(pose_gt_bw) 将真实的位姿 pose_gt_bw 转换成旋转矩阵和平移向量形式。
        # 然后，将平移向量进行归一化处理，并将旋转矩阵和归一化后的平移向量合并为一个新的矩阵 gt_rt。
        # 接下来，定义了一个均方误差损失函数 loss_fn，并设置了其 reduction 参数为 'none'，即不进行降维操作。
        # 然后，通过计算预测的旋转矩阵和平移向量 rot_and_trans 与真实位姿 gt_rt 之间的均方误差损失，得到姿态损失 pose_loss。
        # 在计算损失之前，对前三个元素进行了乘以20的缩放操作。
        # 最后，将姿态损失的均值添加到训练日志中，并将 pose_loss 的均值加到总的损失 loss 中
        if rot_and_trans is not None:  # None
            ##################  Compute Pose Loss ########################
            ### if use deep pose regression
            gt_rt_raw = Pose2RT(pose_gt_bw)
            gt_rt = torch.cat((gt_rt_raw[:, :3], F.normalize(gt_rt_raw[:, 3:])), dim=1)

            loss_fn = torch.nn.MSELoss(reduction='none')
            pose_loss = loss_fn(rot_and_trans, gt_rt).mean(dim=0)
            pose_loss[:3] = pose_loss[:3] * 20
            pose_loss = pose_loss.mean()

            train_writer.add_scalar('pose_loss', pose_loss.mean().item(), n_iter)
            loss = loss + pose_loss.mean()

        # check if there are extreme values
        # 这段代码用于检查深度损失是否超过了极限值。
        # 如果总的损失 loss 大于 9999，则进入调试模式，可以方便地检查代码并进行调试。
        # 这个值是一个比较大的阈值，如果损失超过了这个值，通常意味着训练出了严重的问题，需要进行排查和修复。

        # img = PIL.Image.open(torchvision.transforms.ToPILImage()(input0[0]))
        # img.show()

        if loss > 9999:
            import pdb;
            pdb.set_trace()
            # loss.data = torch.tensor(9999.0, device='cuda:0')

        losses.update(loss.item(), input0.size(0))  # 当前 mini-batch 的损失值 loss 除以 mini-batch 中输入数据的数量 input0.size(0) 并更新平均损失值
        train_writer.add_scalar('train_loss', loss, n_iter)  # 将当前 mini-batch 的损失值写入训练日志中

        # 从优化器对象中获取当前的学习率 cur_lr，并将其写入训练日志中
        cur_lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar('learning_rate', cur_lr, n_iter)

        scaler.scale(loss).backward()  # 然后，通过调用混合精度训练中的 scaler 对象的 scale() 方法，对当前 mini-batch 的损失值进行缩放处理。
        scaler.step(optimizer)  # 接着，调用 backward() 方法计算梯度，并调用 step() 方法更新模型参数。
        scaler.update()  # 最后，调用 update() 方法更新缩放因子，以便在下一个 mini-batch 中使用。

        # 最后，计算当前 mini-batch 的训练时间，并更新训练时间的平均值 batch_time。
        batch_time.update(time.time() - end)
        end = time.time()

        # recording and visualization
        # 这段代码是用于记录和可视化训练过程中的数据和结果。
        # 首先，通过对输入图像进行归一化处理，将其转换为可视化所需的格式，并将其写入训练日志中。
        # 接下来，对深度图进行处理和可视化。将估计的深度图 depth_bw 和真实的深度图 depth_bw_gt 进行处理，以便在可视化时有更好的效果。
        # 根据设定的最小深度值、标签数量和深度图的值，计算出可视化时的深度图。然后，将处理后的深度图写入训练日志中。
        # 如果存在光流 flow_2D_bw，也会对其进行可视化处理，并将可视化结果写入训练日志中。
        # 最后，使用循环遍历每个深度图，并将处理后的深度图和真实深度图写入训练日志中。
        # 同时，还会将当前的训练信息，包括当前的训练轮数、迭代次数、耗时、数据加载耗时、损失等信息，输出到日志中并显示在控制台中。
        if i % args.print_freq == 0:
            input0 = 0.5 + (input0[0]) * 0.5;
            input1 = 0.5 + (input1[0]) * 0.5
            train_writer.add_image(('left' + str(0)), input0, n_iter)
            train_writer.add_image(('right' + str(0)), input1, n_iter)
            ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!! 输出多张图片
            # input0 = 0.5 + (input0) * 0.5;
            # input1 = 0.5 + (input1) * 0.5
            # train_writer.add_images(('left' + str(0)), input0, n_iter)
            # train_writer.add_images(('right' + str(0)), input1, n_iter)

            # for visualization
            disp_bw_init = [cfg.MIN_DEPTH * args.nlabel / (depth_bw_init.detach().cpu())]
            disp_bw = [cfg.MIN_DEPTH * args.nlabel / (depth_bw.detach().cpu())]
            disp_bw_gt = [cfg.MIN_DEPTH * args.nlabel / (depth_bw_gt.squeeze(1).detach().cpu())]

            if flow_2D_bw is not None:
                flo_bw_raw_vis = flow2rgb_raw(flow_2D_bw[0], max_value=128)
                train_writer.add_image(('flo_bw_raw'), flo_bw_raw_vis, n_iter)

            for j in range(len(disp_bw)):
                disp_bw__init_to_show = tensor2array(disp_bw_init[j][0], max_value=80, colormap='bone')
                disp_bw_to_show = tensor2array(disp_bw[j][0], max_value=80, colormap='bone')
                disp_bw_gt_to_show = tensor2array(disp_bw_gt[j][0], max_value=80, colormap='bone')

                train_writer.add_image(('depth_bw_init' + str(j)), disp_bw__init_to_show, n_iter)
                train_writer.add_image(('depth_bw' + str(j)), disp_bw_to_show, n_iter)
                train_writer.add_image(('depth_bw_gt' + str(j)), disp_bw_gt_to_show, n_iter)

            logger.info('Epoch: [{0}][{1}/{2}]\t Batch Time {3}\t Data Time{4}\t Loss {5}'
                        .format(epoch, i, epoch_size, batch_time, data_time, losses))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, scheduler


# 这段代码是一个用于验证模型性能的函数。它接受一个验证数据集的数据加载器（val_loader）、模型（model）、当前的训练轮数（epoch）以及一个可选的日志记录器（logger）作为输入。
def validate(val_loader, model, epoch, logger=None, test_writer=None):
    global args
    # 在函数中，首先定义了一些用于记录性能指标的平均计量器，包括处理每个批次数据的时间、深度估计误差以及一些常见的评价指标，
    # 如平均绝对误差率（abs_rel）、平均平方误差率（sq_rel）、均方根误差（rmse）等。
    batch_time = AverageMeter();  # 在验证集上处理每个批次数据的时间
    depth_EPEs = AverageMeter()  # 深度估计误差

    abs_rel_t = AverageMeter();  # 平均绝对误差率
    sq_rel_t = AverageMeter();  # 平均平方误差率
    rmse_t = AverageMeter();  # 均方根误差
    rmse_log_t = AverageMeter()  # 对数均方根误差
    a1_t = AverageMeter();  # 角度误差（小于 1°、小于 2°、小于 3° 的像素比例）
    a2_t = AverageMeter();
    a3_t = AverageMeter();
    d1_all_t = AverageMeter()  # 深度误差小于 1.25 倍真实深度的像素比例
    l1_inv_t = AverageMeter();  # 逆深度误差
    sc_inv_t = AverageMeter()  # 逆深度误差小于 0.05 的像素比例

    # switch to evaluate mode
    import matplotlib

    model.eval()
    end = time.time()

    dis_list = [];
    rel_dis_list = [];
    scale_list = [];
    change_ratio_list = []
    errors_fw_l = [];
    errors_bw_l = [];
    depth_list = [];
    epe_tmp = []

    # 然后，将模型切换到评估模式，并开始对验证数据集进行迭代。
    # 在每个迭代步骤中，从验证数据集中获取输入图像、相机内参、相机位姿、预测的相机位姿和深度图真值。然后将这些数据移动到设备上进行计算。
    for i, (inputs, intrinsics, poses, pred_poses, depth_gt) in enumerate(val_loader):

        input0 = inputs[0].to(device)
        input1 = inputs[1].to(device)
        pose_gt_fw = poses[0].to(device)
        pose_gt_bw = poses[1].to(device)
        depth_fw_gt = depth_gt[0].to(device)
        depth_bw_gt = depth_gt[1].to(device)
        pred_pose_fw = pred_poses[0].to(device) if pred_poses is not None else None
        pred_pose_bw = pred_poses[1].to(device) if pred_poses is not None else None

        intrinsics_invs = intrinsics.inverse().float().to(device)
        intrinsics = intrinsics.float().to(device)

        raw_shape = input1.shape
        height_raw = raw_shape[2];
        width_raw = raw_shape[3]

        # 接下来，根据具体的业务逻辑，对输入图像进行预处理，如调整尺寸、填充等。
        # if the flow estimation module is not 'DICL', you could modify the codes below accordingly
        # 如果流量估计模块不是“DICL”，则可以相应地修改以下代码
        height_new = int(np.ceil(raw_shape[2] / 128) * 128)
        width_new = int(np.ceil(raw_shape[3] / 128) * 128)
        padding = (0, width_new - raw_shape[3], 0, height_new - raw_shape[2])
        input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)
        input0 = torch.nn.functional.pad(input0, padding, "replicate", 0)

        #######################################
        b, _, h, w = input0.size()
        # 之后，根据模型的类型和需求，进行不同的计算。可能会进行光流估计、位姿估计、深度估计等。
        # 接着，根据具体的评估策略，计算深度估计结果与真实深度之间的误差，并更新计量器。

        # 如果配置参数中设置了RECORD_POSE为True，则会计算位姿估计的误差，并输出到日志中。否则，将输出深度估计的各项评价指标到日志中。
        # 最终，函数返回深度估计误差的平均值。
        # 需要注意的是，这段代码可能依赖于其他模块或函数的实现，因此在理解整个程序的功能时，还需要查看其他部分的代码。
        if cfg.RECORD_POSE:  # False
            # compute forward and backward poses
            pose_fw, flow_2D_fw = model(input0, input1, intrinsics, pose_gt_fw, pred_pose_fw, cfg.GT_POSE, raw_shape[2], raw_shape[3])
            pose_bw, flow_2D_bw = model(input1, input0, intrinsics, pose_gt_bw, pred_pose_bw, cfg.GT_POSE, raw_shape[2], raw_shape[3])

            def pose_to_motion(pose):
                # convert pose matrix to motion
                rt = Pose2RT(pose)
                out = torch.cat((rt[:, :3], F.normalize(rt[:, 3:])), dim=1)
                return out[0].cpu().numpy()

            for bat in range(len(pose_fw)):
                motion_fw = pose_to_motion(pose_fw[bat])
                motion_bw = pose_to_motion(pose_bw[bat])

                motion_gt_fw = pose_to_motion(pose_gt_fw[bat].unsqueeze(0))
                motion_gt_bw = pose_to_motion(pose_gt_bw[bat].unsqueeze(0))

                error_fw = compute_motion_errors(motion_fw, motion_gt_fw, True)
                error_bw = compute_motion_errors(motion_bw, motion_gt_bw, True)

                errors_fw_l.append(np.array(error_fw))
                errors_bw_l.append(np.array(error_bw))

            errors_fw_mean = np.array(errors_fw_l).mean(axis=0)
            logger.info('Pose Error: [{0}/{1}]\t Time {2}\t error rot {3} t {4} trans {5}'.format(i, len(val_loader), batch_time, errors_fw_mean[0], errors_fw_mean[1], errors_fw_mean[2]))
            continue
        else:
            # conduct inference, both depth and pose
            # 进行推理，包括深度和姿势
            flow_2D_bw, pose_bw, depth_bw, time_dict = model(input1, input0, intrinsics, pose_gt_bw, pred_pose_bw, cfg.GT_POSE, raw_shape[2], raw_shape[3])

        if cfg.RESCALE_DEPTH:  # True
            #### could skip this step during inference
            batch_num = len(depth_bw_gt)
            scale = torch.norm(pose_gt_bw[:, :, -1:].squeeze(-1), dim=-1)
            rescale_ratio = scale / cfg.NORM_TARGET
            depth_bw = depth_bw * rescale_ratio.view(batch_num, 1, 1, 1)

        depth_bw = depth_bw[:, :, :height_raw, :width_raw]

        if flow_2D_bw is not None:  # not None
            flow_2D_bw = flow_2D_bw[:, :, :height_raw, :width_raw]
            flow_2D_plot = flow2rgb_raw(flow_2D_bw[0], max_value=128)
            flow_2D_plot = np.transpose(flow_2D_plot, (1, 2, 0))

        ###############################################################
        ### dummy codes
        ### possibly helpful if someone would like to save visualization
        # if args.save_images:
        #     ref_cv =inputs[0][0].cpu().numpy().transpose(1,2,0)[:,:,::-1]
        #     tar_cv =inputs[1][0].cpu().numpy().transpose(1,2,0)[:,:,::-1]
        #     ref_cv = (ref_cv*0.5+0.5)*255;tar_cv = (tar_cv*0.5+0.5)*255
        #     cv2.imwrite(path_1,ref_cv)
        #     realflow_vis = flow_to_image(flow_2D_bw[0].cpu().detach().numpy().transpose(1,2,0),None)
        #     cv2.imwrite(path_2,tar_cv)
        #     cv2.imwrite(path_pred,realflow_vis)
        ###############################################################

        # the same threshold and masking strategy as used by previous methods
        if cfg.DEMON_DATASET:  # False
            mask_bw = (depth_bw_gt <= 10) & (depth_bw_gt >= 0.5) & (depth_bw_gt == depth_bw_gt)
        else:
            mask_bw = (depth_bw_gt > 0) & (depth_bw_gt < 80)
            crop_mask = mask_bw.clone()
            crop_mask[:] = 0
            gt_height, gt_width = mask_bw.shape[2:]
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask[:, :, crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask_bw = mask_bw & crop_mask

        ### median scale shift, because of the scale ambiguity
        median_scale_list = []
        for b_idx in range(len(depth_bw_gt)):
            try:
                cur_median_scale = (depth_bw_gt[b_idx][mask_bw[b_idx]].median() / depth_bw[b_idx][mask_bw[b_idx]].median()).detach();
            except:
                cur_median_scale = depth_bw_gt[b_idx].median() / depth_bw[b_idx].median()
            median_scale_list.append(cur_median_scale)
        median_scale = torch.FloatTensor(median_scale_list).to(depth_bw.device).type_as(depth_bw)
        depth_bw = depth_bw * (median_scale.view(-1, 1, 1, 1))

        # check bound
        max_range = cfg.MIN_DEPTH * args.nlabel
        depth_bw[depth_bw <= cfg.MIN_DEPTH] = cfg.MIN_DEPTH
        depth_bw[depth_bw > max_range] = max_range

        # compute errors
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, l1_inv, sc_inv = evaluate_metric(depth_bw_gt[mask_bw].cpu().numpy(), depth_bw[mask_bw].detach().cpu().numpy())

        # record errors
        abs_rel_t.update(abs_rel, intrinsics.size(0))
        sq_rel_t.update(sq_rel, intrinsics.size(0))
        rmse_t.update(rmse, intrinsics.size(0))
        rmse_log_t.update(rmse_log, intrinsics.size(0))
        a1_t.update(a1, intrinsics.size(0));
        a2_t.update(a2, intrinsics.size(0));
        a3_t.update(a3, intrinsics.size(0))
        l1_inv_t.update(l1_inv, intrinsics.size(0));
        sc_inv_t.update(sc_inv, intrinsics.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # 最后，在每个迭代步骤结束时，记录当前的性能指标，并根据需要保存可视化结果。
        if i % args.print_freq == 0:
            if args.print_freq == 1:
                # print per-frame detail, for debugging
                scale = torch.norm(pose_gt_bw[:, :, -1], dim=-1)
                logger.info('Test: [{0}/{1}]\t Scale {2}\t abs_rel {3} l1_inv {4} sc_inv {5}'.format(i, len(val_loader), scale.mean().item(), abs_rel_t, l1_inv_t, sc_inv_t))

            else:
                logger.info('Test: [{0}/{1}]\t Time {2}\t abs_rel {3}'.format(i, len(val_loader), batch_time, abs_rel_t))

        if args.save_images:
            visual_path = os.path.join(save_path, 'val_visual')
            if not os.path.exists(visual_path):
                os.makedirs(visual_path)
            depth_plot = depth_bw[0].detach().cpu().numpy().squeeze()
            matplotlib.image.imsave(visual_path + "/{0:06}.png".format(i), np.uint16(depth_plot * 256))
            if flow_2D_bw is not None: matplotlib.image.imsave(visual_path + "/{0:06}_flow2d.png".format(i), flow_2D_plot)

    if cfg.RECORD_POSE:
        logger.info('Forward Error:');
        logger.info(np.array(errors_fw_l).mean(axis=0))
        logger.info('Backward Error:');
        logger.info(np.array(errors_bw_l).mean(axis=0))
    else:
        # abs_rel_t.avg：平均绝对误差率
        # sq_rel_t.avg：平均平方误差率
        # rmse_t.avg：均方根误差
        # rmse_log_t.avg：均方根对数误差
        # a1_t.avg、a2_t.avg、a3_t.avg：三个常见评价指标 a1、a2、a3
        # l1_inv_t.avg：逆深度误差
        # sc_inv_t.avg：逆深度误差比例
        # d1_all_t.avg：平均深度误差
        logger.info('abs_rel {0} sq_rel {1} rmse {2} rmse_log {3} a1 {4} a2 {5} a3 {6} l1_inv {7} sc_inv {8} d1_all {9}'.format(abs_rel_t.avg, sq_rel_t.avg, rmse_t.avg, rmse_log_t.avg, a1_t.avg, a2_t.avg, a3_t.avg, l1_inv_t.avg, sc_inv_t.avg, d1_all_t.avg))
    return depth_EPEs.avg


def save_pose(val_loader, model, epoch, logger=None):
    '''
    save sequence pose for evaluation
    '''
    global args
    batch_time = AverageMeter();
    depth_EPEs = AverageMeter()

    abs_rel_t = AverageMeter();
    sq_rel_t = AverageMeter();
    rmse_t = AverageMeter();
    rmse_log_t = AverageMeter()
    a1_t = AverageMeter();
    a2_t = AverageMeter();
    a3_t = AverageMeter()
    l1_inv_t = AverageMeter();
    sc_inv_t = AverageMeter()

    # switch to evaluate mode
    model.eval()
    import matplotlib
    end = time.time()

    dis_list = [];
    rel_dis_list = [];
    scale_list = [];
    change_ratio_list = []

    errors_fw_l = [];
    errors_bw_l = []

    for i, (inputs, intrinsics, poses, pred_poses, depth_gt, img2_path) in enumerate(val_loader):
        input0 = inputs[0].to(device)
        input1 = inputs[1].to(device)
        pose_gt_fw = poses[0].to(device)
        pose_gt_bw = poses[1].to(device)
        depth_fw_gt = depth_gt[0].to(device)
        depth_bw_gt = depth_gt[1].to(device)
        pred_pose_fw = pred_poses[0].to(device) if pred_poses is not None else None
        pred_pose_bw = pred_poses[1].to(device) if pred_poses is not None else None

        intrinsics_invs = intrinsics.inverse().float().to(device)
        intrinsics = intrinsics.float().to(device)

        #######################################
        raw_shape = input1.shape
        height_raw = raw_shape[2];
        width_raw = raw_shape[3]

        # if the flow estimation module is not 'DICL', you could modify the codes below accordingly
        height_new = int(np.ceil(raw_shape[2] / 128) * 128)
        width_new = int(np.ceil(raw_shape[3] / 128) * 128)
        padding = (0, width_new - raw_shape[3], 0, height_new - raw_shape[2])
        input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)
        input0 = torch.nn.functional.pad(input0, padding, "replicate", 0)

        b, _, h, w = input0.size()

        pose_fw, flow_2D_raw_fw = model(input0, input1, intrinsics, pose_gt_fw, pred_pose_fw, cfg.GT_POSE, raw_shape[2], raw_shape[3])
        pose_bw, flow_2D_raw_bw = model(input1, input0, intrinsics, pose_gt_bw, pred_pose_bw, cfg.GT_POSE, raw_shape[2], raw_shape[3])

        # save the predicted poses as a numpy file, each corresponds to a sequence
        for batch_idx in range(len(input0)):
            pred_poses_fb = torch.cat((pose_fw[batch_idx], pose_bw[batch_idx])).cpu().numpy()
            np_save_path = img2_path[batch_idx].replace('.png', '.npy').replace('image_02', 'pred_poses_fb')

            if not os.path.exists(os.path.dirname(np_save_path)):
                os.makedirs(os.path.dirname(np_save_path))
            np.save(np_save_path, pred_poses_fb)
            logger.info('SAVE POSE: [{0}/{1}]\t Time {2}\t '.format(i, len(val_loader), batch_time))
        continue

    if cfg.RECORD_POSE:
        logger.info(cfg.DEMON_DATASET_SPE)
        logger.info('Forward Error:');
        logger.info(np.array(errors_fw_l).mean(axis=0))
        logger.info('Backward Error:');
        logger.info(np.array(errors_bw_l).mean(axis=0))
    else:
        logger.info('abs_rel {0} sq_rel {1} rmse {2} rmse_log {3} a1 {4} a2 {5} a3 {6} l1_inv {7} sc_inv {8}'.format(abs_rel_t.avg, sq_rel_t.avg, rmse_t.avg, rmse_log_t.avg, a1_t.avg, a2_t.avg, a3_t.avg, l1_inv_t.avg, sc_inv_t.avg))

    return depth_EPEs.avg


############################################## Utility ##############################################

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def convert_disps_to_depths_kitti(gt_disp, intrinsic):
    focus = intrinsic[:, 0, 0]
    shape = [*gt_disp.size()]
    mask = (gt_disp > 0).float()
    focus = focus.view(shape[0], 1, 1).repeat(1, shape[1], shape[2]).float()
    gt_depth = (focus * 0.54) / (gt_disp + (1.0 - mask))
    return gt_depth * mask


def evaluate_metric(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    l1_inv = l1_inverse(gt, pred)

    sc_inv = scale_invariant(gt, pred)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, l1_inv, sc_inv


class AverageMeter(object):
    """Computes and stores the average and current value 这是一个用于计算和存储平均值和当前值的辅助类AverageMeter。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 当前值的总和
        self.count = 0  # 当前值的数量

    def update(self, val, n=1):
        self.val = val
        ####### !!!!!!!!!!!!!!!!!!!!!
        if math.isnan(val) or str(self.val) == "nan":
            val = self.avg
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


def flow2rgb_raw(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


if __name__ == '__main__':
    main()
