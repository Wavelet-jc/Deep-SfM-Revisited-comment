from __future__ import print_function

import datetime
import logging
import os

import torch
import numpy as np
import cv2
import utils
import time

from matplotlib import pyplot as plt
from models import DICL_shallow
from models.RAFT.core.raft import RAFT # 一个名为RAFT的模块，可能是用于光流估计任务。

from models import PSNet as PSNet # 一个名为PSNet的模块，可能是用于深度估计的模型。
# import essential_matrix # 进行基础矩阵估计的相关功能
from epipolar_utils import * # 一些用于处理极线几何的工具函数。
from models.PoseNet import ResNet,Bottleneck, PlainPose # 与姿态估计相关的模块。
from lib.config import cfg, cfg_from_file, save_config_to_file

# for speed analysis
global time_dict # 用于存储时间相关的信息
time_dict = {}

try:
    autocast = torch.cuda.amp.autocast # 用于混合精度训练的上下文管理器
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# 这是一个SFMnet模型的定义，用于进行稠密的三维重建。
# 该模型包含了多个组件和模块，包括光流估计器、深度估计器、特征提取器、特征匹配器和姿态估计器等。
class SFMnet(torch.nn.Module):
    # 在初始化函数中，首先定义了一些超参数，包括步长delta、无效像素值alpha、最大迭代次数maxreps、最小匹配点数min_matches、RANSAC算法的迭代次数ransac_iter和判断内点的阈值ransac_threshold等。
    # 然后，根据配置文件中指定的光流估计器和深度估计器的类型，选择相应的模型进行初始化。
    # 其中，光流估计器可以是RAFT模型或DICL_shallow模型，深度估计器可以是PSNet、CVP、PANet、REGNet、REG2D或DISPNET模型。
    # 接下来，定义了SIFT和SURF特征提取器，并使用FLANN匹配器进行特征匹配。
    # 最后，选择姿态估计器，根据配置文件选择使用PlainPose模型或ResNet模型进行初始化。其中，ResNet模型使用了Bottleneck结构。
    def __init__(self, nlabel=64, min_depth=0.5):
        super(SFMnet,self).__init__()
        ##### Hyperparameters #####
        self.delta = 0.001  # 流深度图的步长
        self.alpha = 0.0    # 视差图中默认的无效像素值
        self.maxreps = 200      # 用于光流算法的最大迭代次数
        self.min_matches = cfg.min_matches  # 用于RANSAC算法进行姿态估计时的最小匹配点数 20
        self.ransac_iter = cfg.ransac_iter  # RANSAC算法的迭代次数 5
        self.ransac_threshold = cfg.ransac_threshold    # RANSAC算法中判断是否为内点的阈值。 1e-4

        ##### 模块和组件 #####
        self.nlabel = nlabel    # 视差图中的标签数目。
        self.min_depth = min_depth  # 最小深度值，用于避免数值误差。

        # 选择光流估计器，根据配置文件选择DICL_shallow模型。
        if cfg.FLOW_EST =='RAFT':
            self.flow_estimator = RAFT()
        elif cfg.FLOW_EST =='DICL':  # 用于准确光流估计的位移不变匹配代价学习
            self.flow_estimator = DICL_shallow()
        else:
            raise NotImplementedError

        # 选择深度估计器，根据配置文件选择合适的模型。
        if cfg.DEPTH_EST=='PSNET': # DPSNet End-to-end Deep Plane Sweep Stereo
            self.depth_estimator = PSNet(nlabel,min_depth)
        elif cfg.DEPTH_EST=='CVP':
            from models.CVPMVS import CVPMVS
            self.depth_estimator = CVPMVS()
        elif cfg.DEPTH_EST=='PANET':
            from models.PANet import PANet
            self.depth_estimator = PANet(nlabel,min_depth)
        elif cfg.DEPTH_EST=='REGNET':
            from models.REGNet import REGNet
            self.depth_estimator = REGNet(nlabel,min_depth)
        elif cfg.DEPTH_EST=='REG2D':
            from models.REG2D import REG2D
            self.depth_estimator = REG2D(nlabel,min_depth)
        elif cfg.DEPTH_EST=='DISPNET':
            from models.DISPNET import DISPNET
            self.depth_estimator = DISPNET(nlabel,min_depth)
        else:
            raise NotImplementedError

        # SIFT和SURF特征提取器。
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.surf = cv2.xfeatures2d.SURF_create()

        # FLANN_INDEX_KDTREE是一种用于构建FLANN（Fast Library for Approximate Nearest Neighbors）索引的算法类型。FLANN是一个用于高效近似最近邻搜索的库，它可以在大规模数据集中快速查找最近邻。
        # FLANN_INDEX_KDTREE是基于kd树的一种索引算法。kd树是一种二叉树结构，用于将多维空间划分为不同的区域，以便进行快速的最近邻搜索。它通过递归地选择一个轴，并将数据点按照轴上的值进行划分，构建起一棵树结构。
        # 在FLANN中，FLANN_INDEX_KDTREE算法可以通过构建kd树来加速最近邻搜索。它将数据点存储在kd树中，并使用树的结构来快速定位最近邻。
        # 在代码中，FLANN_INDEX_KDTREE被用作FLANN匹配器的算法参数，用于创建基于kd树的FLANN索引，以便进行特征匹配。
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)  # FLANN匹配器，用于特征匹配

        # 选择姿态估计器，根据配置文件选择使用PlainPose模型或ResNet模型
        if cfg.POSE_EST =='POSENET': # 'RANSAC'
            if cfg.POSE_NET_TYPE  == 'plain':
                self.posenet = PlainPose()
            elif cfg.POSE_NET_TYPE == 'res':
                self.posenet = ResNet(Bottleneck,[3, 4, 6, 3])
            else:
                raise NotImplementedError

    # 模型的前向传播函数forward
    def forward(self, ref, target, intrinsic, pose_gt=None, pred_pose=None, use_gt_pose=False,
                    h_side=None,w_side=None,logger=None,depth_gt=None,img_path=None):

        # if TRAIN_FLOW, we only conduct flow estimation
        # 首先，如果模型处于训练状态且cfg.TRAIN_FLOW为True（即训练流估计），则只进行流估计，并返回流估计结果
        if self.training and cfg.TRAIN_FLOW: # False
            flow_outputs  = self.flow_estimator(torch.cat((ref, target),dim=1))
            return flow_outputs

        # 接下来，将输入的相机内参intrinsic转换为GPU张量。然后根据是否使用真实姿态信息进行处理：
        intrinsic_gpu = intrinsic.float().cuda()                # torch.Size([2, 3, 3])
        intrinsic_inv_gpu = torch.inverse(intrinsic_gpu)        # torch.Size([2, 3, 3])

        # Default, if do not use ground truth poses for training
        # 如果use_gt_pose为False，则根据配置文件选择使用RANSAC算法或POSENET网络进行姿态估计。并返回位姿矩阵P_mat和本质矩阵E_mat。
        if use_gt_pose == False: # False

            # if predict relative poses online, or use pre-saved poses
            if cfg.PRED_POSE_ONLINE:  # 在线预测相对位姿

                # flow estimation
                with autocast(enabled=cfg.MIXED_PREC): # Ture
                    flow_start = time.time()
                    flow_2D, conf  = self.flow_estimator(torch.cat((ref, target),dim=1))    # DICL-flow

                # recover image shape, to avoid meaningless optical flow matches
                if h_side is not None or w_side is not None:
                    flow_2D = flow_2D[:,:,:h_side,:w_side]
                    try:
                        conf = conf[:,:,:h_side,:w_side]
                    except:
                        conf = conf

                # choose how to estimate pose, by RANSAC or deep regression
                if cfg.POSE_EST =='RANSAC':
                    # some inputs are left for possible visualization or debug, plz ignore them if not
                    # return:   Pose matrix             Bx3x4
                    #           Essential matrix        Bx3x3
                    P_mat,E_mat = self.pose_by_ransac(flow_2D,ref,target,intrinsic_inv_gpu,
                                                        h_side,w_side,pose_gt =pose_gt,img_path=img_path)
                    rot_and_trans = None
                elif cfg.POSE_EST =='POSENET':
                    rot_and_trans = self.posenet(flow_2D,conf,ref,target)
                    P_mat = RT2Pose(rot_and_trans)
                else:
                    raise NotImplementedError
            else:
                # use ground truth poses, for oracle experiments
                P_mat = pred_pose; E_mat = None; flow_2D = None

            # if only use gt scales, for oracle experiments                
            if cfg.PRED_POSE_GT_SCALE:   # False
                scale = torch.norm(pose_gt[:,:3, 3],dim=1, p=2).unsqueeze(1).unsqueeze(1)
                P_mat[:,:,-1:] = P_mat[:,:,-1:]*scale

            P_mat.unsqueeze_(1)
        else: # 如果use_gt_pose为True，则直接使用给定的真实姿态信息作为位姿矩阵P_mat。
            E_mat = None
            P_mat = pose_gt.clone()
            if cfg.GT_POSE_NORMALIZED: # 如果配置文件中设置了cfg.GT_POSE_NORMALIZED为True，则对位姿进行归一化处理。
                scale = torch.norm(P_mat[:,:3, 3],dim=1, p=2).unsqueeze(1).unsqueeze(1)
                P_mat[:,:,-1:] = P_mat[:,:,-1:]/scale # 接着，根据配置文件中的设置进行缩放因子的处理，并将P_mat扩展为(batch_size, 1, 3, 4)的维度。
            P_mat.unsqueeze_(1)
            flow_2D = torch.zeros([ref.shape[0],2,ref.shape[2],ref.shape[3]]).cuda().type_as(ref)

        # 如果需要记录位姿信息，则直接返回位姿矩阵P_mat和流估计结果flow_2D。
        if cfg.RECORD_POSE or (cfg.RECORD_POSE_EVAL and not self.training):
            return P_mat, flow_2D

        # 然后，根据输入图像的尺寸进行裁剪操作。
        if h_side is not None or w_side is not None:
            ref = ref[:,:,:h_side,:w_side]; target = target[:,:,:h_side,:w_side]

        # 接下来，通过深度估计器预测深度信息。并返回初始深度图depth_init和修正后的深度图depth。
        # depth prediction
        with autocast(enabled=cfg.MIXED_PREC):
            depth_start = time.time()
            depth_init, depth = self.depth_estimator(ref, [target], P_mat, intrinsic_gpu, intrinsic_inv_gpu,pose_gt=pose_gt,depth_gt=depth_gt,E_mat=E_mat)  # PSNET

        # 最后，根据模型是否处于训练状态进行结果的返回。
        if self.training:
            # rot_and_trans is only used for pose deep regression
            # otherwise, it is None
            return flow_2D, P_mat, depth, depth_init, rot_and_trans
        return flow_2D, P_mat, depth, time_dict


    # 这是一个使用RANSAC算法进行姿态估计的函数。给定输入的光流、参考帧和目标帧，以及相机内参数矩阵的逆矩阵等信息，该函数执行以下步骤：

    # 1 将光流转换为坐标点。
    # 2 检测关键点，并得到关键点的描述子。
    # 3 使用Flann算法进行特征匹配，并筛选出好的匹配点对。
    # 4 根据匹配点对和相机内参数，计算基础矩阵和投影矩阵。
    # 5 循环处理每个batch中的帧，根据配置选择使用SIFT匹配点或光流匹配点。
    # 6 根据相机内参数，将匹配点转换为归一化坐标。
    # 7 使用GPU加速的RANSAC五点算法计算本征矩阵、投影矩阵和基础矩阵。
    # 8 返回投影矩阵和本征矩阵。
    def pose_by_ransac(self, flow_2D, ref, target, intrinsic_inv_gpu,
                            h_side, w_side, pose_gt=False, img_path=None):

        b, _, h, w = flow_2D.size()
        coord1_flow_2D, coord2_flow_2D = flow2coord(flow_2D)    # Bx3x(H*W)
        coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)
        coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)
        margin = 10                 # avoid corner case  避免转角情况


        E_mat = torch.zeros(b, 3, 3).cuda()                     # Bx3x3
        P_mat = torch.zeros(b, 3, 4).cuda()                     # Bx3x4

        PTS1=[]; PTS2=[];                                       # point list

        # process the frames of each batch
        for b_cv in range(b):
            # convert images to cv2 style
            if h_side is not None or w_side is not None:
                ref_cv =ref[b_cv,:,:h_side,:w_side].cpu().numpy().transpose(1,2,0)[:,:,::-1]
                tar_cv =target[b_cv,:,:h_side,:w_side].cpu().numpy().transpose(1,2,0)[:,:,::-1]

            else:
                ref_cv =ref[b_cv].cpu().numpy().transpose(1,2,0)[:,:,::-1]
                tar_cv =target[b_cv].cpu().numpy().transpose(1,2,0)[:,:,::-1]
            ref_cv = (ref_cv*0.5+0.5)*255; tar_cv = (tar_cv*0.5+0.5)*255
            # detect key points           
            kp1, des1 = self.sift.detectAndCompute(ref_cv.astype(np.uint8),None)
            kp2, des2 = self.sift.detectAndCompute(tar_cv.astype(np.uint8),None)

            ################ debug ######################
            # folder_path = "output/sift/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
            # os.makedirs(folder_path)
            # img = None
            # img = cv2.drawKeypoints(ref_cv.astype(np.uint8), kp1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # file_path = os.path.join(folder_path, 'sift_keypoints1.jpg')
            # cv2.imwrite(file_path, img)
            # img = cv2.drawKeypoints(tar_cv.astype(np.uint8), kp2, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # file_path = os.path.join(folder_path, 'sift_keypoints2.jpg')
            # cv2.imwrite(file_path, img)
            ################ debug ######################

            if len(kp1)<self.min_matches or len(kp2)<self.min_matches:
                # surf generally has more kps than sift
                kp1, des1 = self.surf.detectAndCompute(ref_cv.astype(np.uint8),None)
                kp2, des2 = self.surf.detectAndCompute(tar_cv.astype(np.uint8),None)

            try:
                # filter out some key points
                matches = self.flann.knnMatch(des1,des2,k=2)
                good = []; pts1 = []; pts2 = []
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.5*n.distance:  ### 0.8 !!!!!!!!!!!!!!!!
                        good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)

                # 如果好的匹配数目不够，则不进行筛选，全部都保留
                # degengrade if not existing good matches
                if len(good)<self.min_matches:
                    good = [];pts1 = [];pts2 = []
                    for i,(m,n) in enumerate(matches):
                        good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
                pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2);

                ################ debug ######################
                ### 显示保存匹配图片
                # img3 = cv2.drawMatchesKnn(ref_cv.astype(np.uint8), kp1, tar_cv.astype(np.uint8), kp2, matches, None)
                # file_path = os.path.join(folder_path, 'sift_drawMatches.jpg')
                # cv2.imwrite(file_path, img3)
                ################ debug ######################
            except:
                # if cannot find corresponding pairs, ignore this sift mask 
                PTS1.append([None]); PTS2.append([None])


        assert len(PTS1)==b

        for batch in range(b):
            if cfg.SIFT_POSE:   # False
                # if directly use SIFT matches
                pts1 = PTS1[batch]; pts2 = PTS2[batch]
                coord1_sift_2D = torch.FloatTensor(pts1)
                coord2_sift_2D = torch.FloatTensor(pts2)
                coord1_flow_2D_norm_i = torch.cat((coord1_sift_2D,torch.ones(len(coord1_sift_2D),1)),dim=1).unsqueeze(0).to(coord1_flow_2D.device).permute(0,2,1)
                coord2_flow_2D_norm_i = torch.cat((coord2_sift_2D,torch.ones(len(coord2_sift_2D),1)),dim=1).unsqueeze(0).to(coord1_flow_2D.device).permute(0,2,1)
            else:
                # check the number of matches(corresponding pairs)
                if len(PTS1[batch])<self.min_matches or len(PTS2[batch])<self.min_matches:
                    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,margin:-margin,margin:-margin].contiguous().view(3,-1).unsqueeze(0)
                    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,margin:-margin,margin:-margin].contiguous().view(3,-1).unsqueeze(0)
                else:
                    if cfg.SAMPLE_SP:  # False
                        # conduct interpolation
                        pts1 = torch.from_numpy(PTS1[batch]).to(coord1_flow_2D.device).type_as(coord1_flow_2D)
                        B, C, H, W = coord1_flow_2D.size()
                        pts1[:,0] = 2.0*pts1[:,0]/max(W-1,1)-1.0;pts1[:,1] = 2.0*pts1[:,1]/max(H-1,1)-1.0
                        coord1_flow_2D_norm_i = F.grid_sample(coord1_flow_2D[batch].unsqueeze(0), pts1.unsqueeze(0).unsqueeze(-2),align_corners=True).squeeze(-1)
                        coord2_flow_2D_norm_i = F.grid_sample(coord2_flow_2D[batch].unsqueeze(0), pts1.unsqueeze(0).unsqueeze(-2),align_corners=True).squeeze(-1)
                    else:
                        # default choice
                        # 默认选择：将 PTS1[batch] 转换为整数型数组，并从 coord1_flow_2D 和 coord2_flow_2D 中提取出相应的数据。
                        pts1 = np.int32(np.round(PTS1[batch]))
                        coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
                        coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

            intrinsic_inv_gpu_i = intrinsic_inv_gpu[batch].unsqueeze(0) # 将 intrinsic_inv_gpu 中当前批次的数据提取出来，并进行一些形状操作

            # projection by intrinsic matrix
            # 将二维坐标张量 coord1_flow_2D_norm_i 和 coord2_flow_2D_norm_i 投影到相机坐标系中。
            # 具体来说，它使用逆相机内参矩阵 intrinsic_inv_gpu_i 将二维坐标转换为三维坐标，以便后续的处理。
            # bmm 函数（（batch matrix multiplication），该函数用于执行批次矩阵乘法
            coord1_flow_2D_norm_i = torch.bmm(intrinsic_inv_gpu_i, coord1_flow_2D_norm_i)
            coord2_flow_2D_norm_i = torch.bmm(intrinsic_inv_gpu_i, coord2_flow_2D_norm_i)
            # reshape coordinates
            # 具体来说，它首先使用 transpose 函数将 coord1_flow_2D_norm_i 和 coord2_flow_2D_norm_i 的第二维和第三维进行转置。
            # 然后，它使用索引 [0,:,:2] 从中提取出第一个批次的数据，并去除最后一维（即保留前两维），得到形状为 (num_points, 2) 的张量。
            # 最后，它使用 contiguous 函数将张量在内存中重新排列，以便后续的计算。
            coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.transpose(1,2)[0,:,:2].contiguous()
            coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.transpose(1,2)[0,:,:2].contiguous()

            with autocast(enabled=False):
                # GPU-accelerated RANSAC five-point algorithm
                # 这段代码的作用是使用 GPU 加速的 RANSAC 五点算法计算本征矩阵 E_i、投影矩阵 P_i、基础矩阵 F_i 和内点数量 inlier_num。
                # 该函数使用 RANSAC 五点算法从二维坐标张量 coord1_flow_2D_norm_i 和 coord2_flow_2D_norm_i 中估计本质矩阵 E_i 和基础矩阵 F_i，并计算投影矩阵 P_i 和内点数量 inlier_num
                # 数还使用相机内参矩阵 intrinsic_inv_gpu[batch,:,:] 对坐标进行校正，并传递一些其他参数（如阈值、迭代次数等）。
                E_i, P_i, F_i,inlier_num = compute_P_matrix_ransac(coord1_flow_2D_norm_i.detach(), coord2_flow_2D_norm_i.detach(),
                                                                intrinsic_inv_gpu[batch,:,:], self.delta, self.alpha, self.maxreps,
                                                                len(coord1_flow_2D_norm_i), len(coord1_flow_2D_norm_i),
                                                                self.ransac_iter, self.ransac_threshold) # 5, 0.0001
            # 最后，它将计算得到的 E_i 和 P_i 分别存储在 E_mat 和 P_mat 张量中的对应位置。
            # 由于这里使用了 detach() 函数，因此计算结果不会对计算图产生影响，这意味着这些张量不会被用于反向传播。
            E_mat[batch, :, :] = E_i.detach(); P_mat[batch, :, :] = P_i.detach()

        return P_mat, E_mat




#############################################  Utility #############################################

def check_tensor(tensor):
    return torch.isinf(tensor).any() or torch.isnan(tensor).any()

# 将位姿矩阵转换为旋转和平移向量
def Pose2RT(pose_mat):
    # pose_mat [B,3,4]
    # return : (d1,d2,d3,t1,t2,t3)
    cur_angle = utils.matrix2angle(pose_mat[:,:3,:3])
    cur_trans = pose_mat[:,:3,-1]
    return torch.cat((cur_angle,cur_trans),dim=-1)

# 将旋转和平移向量转换为位姿矩阵
def RT2Pose(RT):
    # RT (d1,d2,d3,t1,t2,t3)
    # return : [B,3,4]
    cur_rot = utils.angle2matrix(RT[:,:3])
    cur_trans = RT[:,3:].unsqueeze(-1)
    return torch.cat((cur_rot,cur_trans),dim=-1)

# 用于将光流图(flow)转换为坐标点对(coord1_hom, coord2_hom)的函数
def flow2coord(flow):
    """
    Generate flat homogeneous coordinates 1 and 2 from optical flow. 
    Args:
        flow: bx2xhxw, torch.float32
    Output:
        coord1_hom: bx3x(h*w)
        coord2_hom: bx3x(h*w)
    """
    b, _, h, w = flow.size() # 输入参数flow是一个4D张量，大小为bx2xhxw，其中b表示batch大小，h和w分别表示图像的高度和宽度
    coord1 = torch.zeros_like(flow) # 根据光流图的大小创建一个与其相同大小的全零张量coord1，并在每个通道上添加对应的x坐标和y坐标。
    coord1[:,0,:,:] += torch.arange(w).float().cuda()
    coord1[:,1,:,:] += torch.arange(h).float().cuda()[:, None]
    coord2 = coord1 + flow      # 将coord1与光流图相加得到coord2，表示每个像素点在参考图像和目标图像之间的对应坐标。
    coord1_flat = coord1.reshape(b, 2, h*w) # 将coord1和coord2重新调整形状为bx2x(h*w)，得到coord1_flat和coord2_flat，分别表示扁平化后的坐标点对
    coord2_flat = coord2.reshape(b, 2, h*w) #

    ones = torch.ones((b, 1, h*w), dtype=torch.float32).cuda() # 创建一个全1的张量ones，大小为(b, 1, h*w)，表示齐次坐标的第三个分量
    coord1_hom = torch.cat((coord1_flat, ones), dim=1)
    coord2_hom = torch.cat((coord2_flat, ones), dim=1)
    return coord1_hom, coord2_hom

def coord2flow(coord1, coord2, b, h, w):
    """
    Convert flat homogeneous coordinates 1 and 2 to optical flow. 
    Args:
        coord1: bx3x(h*w)
        coord2: bx3x(h*w)
    Output:
        flow: bx2xhxw, torch.float32
    """
    coord1 = coord1[:, :2, :] # bx2x(h*w)
    coord2 = coord2[:, :2, :] # bx2x(h*w)
    flow = coord2 - coord1
    flow = flow.reshape(b, 2, h, w)
    return flow

