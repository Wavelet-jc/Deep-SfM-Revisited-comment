import logging

import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
import glob
from pdb import set_trace as st
import time
import cv2
from lib.config import cfg, cfg_from_file, save_config_to_file
from torchvision.transforms import ColorJitter
from PIL import Image
from utils import kitti_readlines,read_calib_file
import utils
import random
from kitti_utils import generate_depth_map



def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(png_path, -1)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0
    return(flo_img)

# 这段代码是一个用于加载视差图像（disparity map）的函数 disparity_loader_png。它接受一个路径作为输入参数，并返回加载后的视差图像。
# 函数首先判断给定的路径是否存在。如果路径存在，它会使用 OpenCV 的 imread 函数读取图像文件，并将其保存在变量 disp_file 中。参数 -1 表示以原始的 16 位深度读取图像，保留原始位深度信息。
# 然后，将 disp_file 转换为浮点数类型的视差图像，并保存在变量 disp 中。这里假设视差值已经归一化到 [0, 1] 的范围内。因此，代码中将 disp 除以 256 进行归一化处理。
# 最后，通过添加一个额外的维度，将 disp 扩展为三维数组，以匹配典型的视差图像形状。
# 如果给定的路径不存在，那么将返回 None。
# 请注意，该代码中引用了 os、cv2 和 np 模块，因此在使用之前需要确保这些模块已经正确导入。
def disparity_loader_png(path):
    if os.path.exists(path):
        disp_file = cv2.imread(path, -1)
        disp = disp_file.astype(np.float32)
        # 将disp除以256，将视差值缩放到0到1之间的浮点数范围。
        # 这是因为视差图像通常使用16位无符号整数表示，范围在0到65535之间，而将其除以256可以将其缩放到0到255之间。
        disp = disp / 256
        disp = np.expand_dims(disp, 2)

    else:
        disp = None
    return disp

def mask_loader_png(path):
    if os.path.exists(path):
        disp_file= cv2.imread(path, -1)
        disp_file = disp_file > 0
        disp = disp_file.astype(np.float32)
        disp = np.expand_dims(disp, 2)
    else:
        disp = None
    return disp

def load_intrinsics(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        data = {}
        for line in lines:
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
        p_mat = np.resize(data['P2'], (3, 4))
        intrinsics = p_mat[:,:3]
    return intrinsics

def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as readfile:
        lines = readfile.readlines()
        for line in lines:
            line = line.strip()
            pose = np.fromstring(line, dtype=float, sep=' ')
            pose = pose.reshape(3, 4)
            #pose = np.vstack((pose, [0, 0, 0, 1]))
            pose = pose.astype(np.float32)
            poses.append(pose)
        return poses

def compute_deltaRT(ps1,ps2):   
    R1 = ps1[:3,:3]
    T1 = ps1[:,3:]  
    R2 = ps2[:3,:3]
    T2 = ps2[:,3:]
    Rf = R1.T.dot(R2)
    Tf = R1.T.dot(T2-T1)
    pose = np.concatenate((Rf, Tf), axis=1)
    return pose



class KITTIVOLoaderGT(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, co_transform=None, train=True):
        self.root = root

        # sequence for training and testing
        self.train_seqs = [0,1,2,3,4,5,6,7,8]
        self.test_seqs = [9,10]
        
        self.train = train

        if train:
            self._collect_train_frames()
            self._collect_train_frames_gt()
            self.path_list = self.train_frames
            self.path_list_gt = self.train_frames_gt
            self.path_list_gt_mask = self.train_frames_gt_mask
        else:
            self._collect_test_frames()
            self._collect_test_frames_gt()
            self.path_list = self.test_frames
            self.path_list_gt = self.test_frames_gt
            self.path_list_gt_mask = self.test_frames_gt_mask

        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.pose_dict = {i:load_poses(os.path.join(self.root, 'poses', '{:02d}.txt'.format(i))) for i in range(11)}
        self.photo_aug = ColorJitter.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
        self.asymmetric_color_aug_prob = 0.2

        # pred_poses will work if PRED_POSE_ONLINE = True
        # in order to reduce time cost during training
        # you should first save the predicted poses for each sequence
        try:
            self.pred_poses_fw = {i: np.load(os.path.join(self.root, 'pred_poses','{:02d}_fw.npy').format(i)) for i in range(11)}
            self.pred_poses_bw = {i: np.load(os.path.join(self.root, 'pred_poses','{:02d}_bw.npy').format(i)) for i in range(11)}
        except:
            print('Do not have pre-set relative poses')

    def _collect_train_frames(self):
        self.train_frames = []
        self.seq_len = []
        for seq in self.train_seqs:
            img_dir = os.path.join(self.root, "sequences", "{:02d}".format(seq), "image_2")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            N = len(img_paths)
            self.train_frames.extend(img_paths)
            self.seq_len.append(N)
        self.train_frames = sorted(self.train_frames)

    def _collect_train_frames_gt(self):
        self.train_frames_gt = []
        self.train_frames_gt_mask = []
        for seq in self.train_seqs:
            # you could change img_dir to any other data sources,
            # if do not want to use KITTI original GT
            img_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")
            mask_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            mask_paths = glob.glob(os.path.join(mask_dir, '*.png'))
            self.train_frames_gt.extend(img_paths)
            self.train_frames_gt_mask.extend(mask_paths)
        self.train_frames_gt = sorted(self.train_frames_gt)
        self.train_frames_gt_mask = sorted(self.train_frames_gt_mask)

    def _collect_test_frames(self):
        self.test_frames = []
        self.seq_len = []
        for seq in self.test_seqs:
            img_dir = os.path.join(self.root, "sequences", "{:02d}".format(seq), "image_2")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            N = len(img_paths)
            self.test_frames.extend(img_paths)
            self.seq_len.append(N)
        self.test_frames = sorted(self.test_frames)

    def _collect_test_frames_gt(self):
        self.test_frames_gt = []
        self.test_frames_gt_mask = []
        for seq in self.test_seqs:
            # you could change img_dir to any other data sources,
            # if do not want to use KITTI original GT
            img_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")
            mask_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            mask_paths = glob.glob(os.path.join(mask_dir, '*.png'))
            self.test_frames_gt.extend(img_paths)
            self.test_frames_gt_mask.extend(mask_paths)
        self.test_frames_gt = sorted(self.test_frames_gt)
        self.test_frames_gt_mask = sorted(self.test_frames_gt_mask)


    def __getitem__(self, index):
        # load gt
        gt1_path = self.path_list_gt[index]
        gt1_path_mask = self.path_list_gt_mask[index]

        # load image
        img1_path = self.path_list[index]
        path1_split = img1_path.split('/')
        seq_1 = int(path1_split[-3])
        img_id_1 = int(os.path.splitext(path1_split[-1])[0])
        skip = cfg.SKIP
        
        try:
            img2_path = self.path_list[index+skip]
            gt2_path = self.path_list_gt[index+skip]
            gt2_path_mask = self.path_list_gt_mask[index+skip]
        except:
            img2_path = self.path_list[index-skip]
            gt2_path = self.path_list_gt[index-skip]
            gt2_path_mask = self.path_list_gt_mask[index-skip]

        path2_split = img2_path.split('/')
        seq_2 = int(path2_split[-3])
        img_id_2 = int(os.path.splitext(path2_split[-1])[0])

        if seq_1 != seq_2:
            img2_path = self.path_list[index-skip]
            gt2_path = self.path_list_gt[index-skip]
            gt2_path_mask = self.path_list_gt_mask[index-skip]
            path2_split = img2_path.split('/')
            seq_2 = int(path2_split[-3])
            img_id_2 = int(os.path.splitext(path2_split[-1])[0])

        assert(seq_1 == seq_2)

        inputs = [img1_path, img2_path] 
        gt_depth = [gt1_path,gt2_path]
        gt_depth_mask = [gt1_path_mask,gt2_path_mask]

        # load intrinsic
        calib = os.path.join(self.root, "sequences", "{:02d}".format(seq_1), "calib.txt")

        pose_1 = self.pose_dict[seq_1][img_id_1]
        pose_2 = self.pose_dict[seq_2][img_id_2]
        pose_bw = compute_deltaRT(pose_1,pose_2)
        pose_fw = compute_deltaRT(pose_2,pose_1)
        poses = [pose_fw, pose_bw]

        # pred_poses will work if PRED_POSE_ONLINE = True
        # in order to reduce time cost during training
        # you should first save the predicted poses for each sequence
        try:
            pred_pose_fw = self.pred_poses_fw[seq_1][img_id_1]
            pred_pose_bw = self.pred_poses_bw[seq_1][img_id_1]
            pred_poses = [pred_pose_fw, pred_pose_bw]
        except:
            # just Placeholder
            pred_poses = [pose_fw*0, pose_bw*0]

        # write load images and intrinsics 
        imgs = [os.path.join(self.root, path) for path in inputs]
        intrinsic = os.path.join(self.root, calib)

        depth_gt_mask = [mask_loader_png(gt_mask) for gt_mask in gt_depth_mask]
        depth_gt = [disparity_loader_png(gt) for gt in gt_depth]
        depth_gt = [a*b for a,b in zip(depth_gt,depth_gt_mask)]
        inputs, depth_gt, calib = [cv2.imread(img)[:,:,::-1].astype(np.uint8) for img in imgs], depth_gt, load_intrinsics(intrinsic)

        if self.train:
            if random.random() > 0.5:
                image_stack = np.concatenate([inputs[0], inputs[1]], axis=0)
                image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
                img1, img2 = np.split(image_stack, 2, axis=0)
                inputs[0] = img1.astype(np.float32);
                inputs[1] = img2.astype(np.float32)

        if self.co_transform is not None:
            inputs, depth_gt, calib = self.co_transform(inputs, depth_gt, calib)
        if self.transform is not None:
            inputs = self.transform(inputs)  
        if self.target_transform is not None:
            depth_gt = self.target_transform(depth_gt)

        return inputs, calib, poses,pred_poses, depth_gt, img1_path,img2_path

    def __len__(self):
        return len(self.path_list)



class KITTIRAWLoaderGT(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, co_transform=None, train=True):

        self.root = root

        # you could project velodyne_points to depth png maps and save them to gt_depth_dir,
        # or use KITTI official depth maps, 
        # refer to L362 for more details 

        self.gt_depth_dir = cfg.GT_DEPTH_DIR  # use KITTI official depth maps

        self.train = train

        train_files = os.path.join(self.root, 'train_files.txt')

        if cfg.KITTI_697:  ### Eigen Split KITTI_697: True
            # 697 samples
            test_files = os.path.join(self.root, 'test_files.txt')
        else:
            # 652 samples
            test_files = os.path.join(self.root, 'test_files_benchmark.txt') 

        if self.train:
            self.path_list = kitti_readlines(train_files)
        else:
            self.path_list = kitti_readlines(test_files)
            if cfg.EIGEN_SFM:  # False
                # filter 256 samples from 652
                assert (not cfg.KITTI_697)
                eigen_filter_mask = np.load(os.path.join(self.root, 'eigen_sfm_mask.npy'))
                eigen_filter_idx = np.array(np.nonzero(eigen_filter_mask))[0]
                self.path_list = np.array(self.path_list)[eigen_filter_idx]
                
        self.calib_dict = np.load(os.path.join(self.root, 'kitti_raw_calib_dict.npy'), allow_pickle=True).item()
        self.pose_dict = np.load(os.path.join(self.root, 'kitti_raw_pose_dict.npy'), allow_pickle=True).item()

        # 将数据转换函数、目标转换函数、协同转换函数和颜色增强参数进行初始化
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        fn_idx, b, c, s, h = ColorJitter.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (0, 0.1))
        self.colorJitter = ColorJitter(b, c, s, h)

        if cfg.FILTERED_PAIR and not self.train:
            if cfg.KITTI_697:
                self.img1_path_list = np.load(os.path.join(self.root,'val_img1_list_697.npy'),allow_pickle=True)
            else:
                self.img1_path_list = np.load(os.path.join(self.root,'val_img1_list_652.npy'),allow_pickle=True)
                if cfg.EIGEN_FILTER:
                    self.img1_path_list = np.load(os.path.join(self.root,'val_img1_list_256.npy'),allow_pickle=True)
                    self.img1_path_list = self.img1_path_list[eigen_filter_idx]


    def __getitem__(self, index):
        # 首先从路径列表中获取文件夹路径、帧ID，并将帧ID转换为整数类型。然后根据配置参数随机选择样本的偏移量。
        # 接着根据文件夹路径和帧ID构建参考帧和目标帧的图像路径。
        # 如果设置了过滤对并且当前不是训练模式，则使用预先计算好的目标帧图像路径。
        # 否则，根据偏移量计算目标帧图像路径，如果路径不存在，则使用相反的偏移量计算路径。
        folder, frame_id_2, _ = self.path_list[index].split()   # '2011_09_26/2011_09_26_drive_0051_sync 120 r'
        frame_id_2 = int(frame_id_2)

        # randomly pick samples to build training pairs
        # 随机偏移选取目标样本构建训练对
        offset = -1
        if cfg.RANDOM_OFFSET and random.random()>0.7:
            offset = -2
        if cfg.RANDOM_FW_BW and random.random()>0.5:
            offset = -offset

        time_name = os.path.basename(os.path.dirname(folder))   # e.g.‘2011_09_26’
        calib = self.calib_dict[time_name]  # calib 文件数据

        # reference frame
        img2_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2))

        # target frame
        if cfg.FILTERED_PAIR and (not self.train):
            img1_path = os.path.join(self.root, self.img1_path_list[index])
        else:
            img1_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2+offset))
            if not os.path.exists(img1_path):
                img1_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2-offset))

        frame_id_1 = int(os.path.splitext(os.path.basename(img1_path))[0])

        # 接下来，根据图像路径获取帧ID，并从姿态字典中获取对应的姿态矩阵。
        # 计算参考帧和目标帧之间的前向和后向相对姿态矩阵，并将它们转换为np.float32类型。
        seq_pose = self.pose_dict[os.path.basename(folder)]
        pose_1 = seq_pose[frame_id_1][:3,:]  # target frame
        pose_2 = seq_pose[frame_id_2][:3,:]  # reference frame
        pose_fw = compute_deltaRT(pose_2,pose_1).astype(np.float32)     # 计算从 reference frame 到 target frame 的相对运动
        pose_bw = compute_deltaRT(pose_1,pose_2).astype(np.float32)     # 从 target frame 到reference frame 的相对运动
        poses = [pose_fw, pose_bw]

        # you could save predicted poses to reduce time cost during training
        # please specify your path correspondingly
        # 你可以加载预测的姿势，以减少训练中的时间成本，请相应地指定你的路径
        try:
            pred_poses = np.load(img2_path.replace('image_02','pred_poses_fb').replace('png','npy'))
            pred_poses = [pred_poses[0],pred_poses[1]]
        except:
            # just Placeholder
            pred_poses = [pose_fw*0, pose_bw*0]
        # 然后将参考帧图像2和目标帧图像1读取为np.uint8类型的三通道数组，并存储在列表inputs中。
        inputs = [img1_path, img2_path]
        inputs = [cv2.imread(img)[:,:,::-1].astype(np.uint8) for img in inputs]


        ###################################################################
        ### Please check your address correspondingly
        # 需要估计target图像1的深度图，所以不需要获取图像1的深度图
        if self.train:
            # gt_depth1_path = os.path.join(self.gt_depth_dir, 'train', os.path.basename(folder),'proj_depth/groundtruth/image_02','{:010d}.png'.format(frame_id_1))
            gt_depth2_path = os.path.join(self.gt_depth_dir, 'train', os.path.basename(folder),'proj_depth/groundtruth/image_02','{:010d}.png'.format(frame_id_2))
        else:
            # gt_depth1_path = os.path.join(self.gt_depth_dir, 'val', os.path.basename(folder),'proj_depth/groundtruth/image_02','{:010d}.png'.format(frame_id_1))
            gt_depth2_path = os.path.join( self.gt_depth_dir, 'val', os.path.basename(folder), 'proj_depth/groundtruth/image_02', '{:010d}.png'.format(frame_id_2))

        # 如果不存在深度图，则从点云文件中获取数据进行计算深度
        if not os.path.exists(gt_depth2_path):
            calib_dir = os.path.join(self.root, folder.split("/")[0])
            velo_filename = os.path.join(self.root, folder, "velodyne_points/data", "{:010d}.bin".format(frame_id_2))
            gt_depth2 = generate_depth_map(calib_dir, velo_filename, 2, True)
            gt_depth2 = np.expand_dims(gt_depth2,2).astype(np.float32)
        else:
            gt_depth2 = disparity_loader_png(gt_depth2_path)

        # gt_depth1 is only a placeholder here; you could set any other things
        # gt_depth1在这里只是一个占位符；你可以设置任何其他东西
        gt_depth1 = gt_depth2.copy()
        depth_gt = [gt_depth1, gt_depth2]   # 保存的是视差图, 归一化

        ####### 数据增强操作: 随机水平翻转，随机颜色变换 #######
        # 数据加载器类中的一部分，用于在训练模式下进行数据增强操作
        # cv2.namedWindow("-1")
        # cv2.imshow("-1", inputs[0])
        ## !!!!!!!!!!!!!!!!!!!!!!!!!!1
        if self.train:
            # 根据cfg.FLIP_AUG配置参数确定是否进行水平翻转增强。
            if cfg.FLIP_AUG:
                if random.random() > 0.75:
                    inputs[0] = np.flip(inputs[0],axis=1); inputs[1] = np.flip(inputs[1],axis=1)
                    depth_gt[0] = np.flip(depth_gt[0],axis=1); depth_gt[1] = np.flip(depth_gt[1],axis=1)
        #
        #     # 根据随机生成的一个0到1之间的数大于0.5的概率，进行图像增强操作。
        #     # 首先，将参考帧和目标帧沿着垂直方向合并为一个图像堆叠。
        #     # 然后，将堆叠后的图像转换为PIL.Image类型，并使用self.photo_aug函数对图像进行增强。
        #     # ## 进过颜色变换输出的图片有问题  !!!!!
        #     if random.random() > 0.5:
        #         # print(self.1)
        #         image_stack = np.concatenate([inputs[0], inputs[1]], axis=0)
        #         image_stack = np.array(self.colorJitter.forward(Image.fromarray(image_stack)), dtype=np.uint8)
        #         img1, img2 = np.split(image_stack, 2, axis=0)
        #         inputs[0] = img1.astype(np.float32); inputs[1] = img2.astype(np.float32)
        #         cv2.namedWindow("0")
        #         cv2.imshow("0", inputs[0])

        if self.co_transform is not None:
            # cv2.namedWindow("1")
            # cv2.imshow("1", inputs[0])
            inputs, depth_gt, calib = self.co_transform(inputs, depth_gt, calib)  # 裁剪至(376, 1241, 3)  (376, 1241, 1) (3, 3)
            # cv2.namedWindow("2")
            # cv2.imshow("2", inputs[0])
            cv2.waitKey()

        if self.transform is not None:  # input_transform 使用ArrayToTensorCo将图像数组转换为张量，然后使用NormalizeCo进行归一化操作。
            inputs = self.transform(inputs)

        if self.target_transform is not None:   # 将图像数组转换为张量
            depth_gt = self.target_transform(depth_gt)

        if cfg.SAVE_POSE:
            return inputs, calib, poses, pred_poses, depth_gt,img2_path

        if cfg.GENERATE_KITTI_POSE_TO_SAVE:
            return inputs, calib, poses, depth_gt, seq_1, img_id_1
        else:
            return inputs, calib, poses, pred_poses, depth_gt  # 输入2张图(target,reference)， 校准(3, 3)， 前向相对位姿和反向相对位姿， 预测位姿为空(just Placeholder)， 视差图



    def __len__(self):
        return len(self.path_list)
