import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

import numpy as np
import torch.nn.functional as F

import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
import gc
##編集ファイル
#・bev_backbone
#・transFusion_head，model/init
#・detector3d_template

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    
    parser.add_argument('--Hmodel', type=str, default=None, help='model that outputs a heatmap mode')
    parser.add_argument('--bp_scholar', type=int, default=None, help='selsect backprop scholar')
    parser.add_argument('--grad_layer', type=str, default=None, help='selsect grad_layer')
    parser.add_argument('--layer_number', type=int, default=0, help='selsect grad_layer')
    parser.add_argument('--save_file_name', type=str, default='./layer_results', help='selsect grad_layer')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def project(point,lidar2image):
    # box_3d = box_3d.detach().numpy()
    lidar2image = lidar2image.detach().numpy()
    # _,_,box_corner = translate_boxes_to_open3d_instance(box_3d)

    # lidar2camera_r = np.linalg.inv(input_dict["sensor2lidar_rotation"])
    # lidar2camera_t = (
    #     input_dict["sensor2lidar_translation"] @ lidar2camera_r.T
    # )
    # lidar2camera_rt = np.eye(4).astype(np.float32)
    # lidar2camera_rt[:3, :3] = lidar2camera_r.T
    # lidar2camera_rt[3, :3] = -lidar2camera_t
    # camera_intrinsics = np.eye(4).astype(np.float32)
    # camera_intrinsics[:3, :3] = input_dict["camera_intrinsics"]
    # lidar2image = camera_intrinsics @ lidar2camera_rt.T
   
    cur_coords = np.array(point)
    cur_coords = lidar2image[:3, :3] @ cur_coords.T
    cur_coords += lidar2image[:3, 3].reshape((3, 1))

    # # 画像座標での座標値
    dists = np.clip(cur_coords[2], 1e-5, 1e5)
    x_coords = ((cur_coords[0] / dists ) * 0.48)-32
    y_coords = (((cur_coords[1] / dists )-70) * 0.48)-176

    return x_coords,y_coords

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    #学習に使用する最適化手法の定義(本来はここで定義しない)
    model.cuda()
    model = model.eval() #評価モードへ変更
    
    #学習済みモデルの読み込み
    chk_pt_kitti = torch.load(args.ckpt)

    #学習済みの重みをモデルへ移行
    model.load_state_dict(chk_pt_kitti['model_state'])

    #出力される画像と点群データの保存フォルダを作成

    # integ_dir = "./pred_pcds_f"
    # pred4_dir = args.save_file_name#"/pred_pcdV4"
    # model_result_dir = ""#"/" + args.Hmodel
    # save_dir = integ_dir + pred4_dir + model_result_dir
    # if os.path.exists(integ_dir + pred4_dir + model_result_dir):
    #     shutil.rmtree(integ_dir + pred4_dir + model_result_dir)
    # else:
    #     pass

    # os.makedirs(integ_dir + pred4_dir + model_result_dir, exist_ok=True)

    #入力データの定義
    in_iter = iter(test_loader)

    #可視化する対象を決定
    point_on = True
    gt_box_on = True
    pred_box_on = True

    #可視化するデータを決定
    #in_data_num = '30901'
    # in_data_num = '000211'
    #in_data_num = '001714'
    for i in tqdm(range(len(in_iter)), desc="PC_Processing", total=len(in_iter), ncols=100):
        # if i < 40:
        #     continue
        in_data = next(in_iter)
        invalid_ids = [
            "1000410", "1000420", "1000430", "1000440", "1000450", "1000469", "1000474",
            "1000885", "1000895", "1000905", "1000915", "1000925", "1000935", "1000945",
            "1000955", "1001370", "1001380", "1001390", "1001400", "1001410", "1001420",
            "1001430", "1001440"
        ]
        ####nuScenes用
        # if in_data["scene_name_info"] != "scene-0638" and in_data["scene_name_info"] != "scene-1066"  and in_data["scene_name_info"] != "scene-0520" and in_data["scene_name_info"] != "scene-0108":
        #     continue
        #####
        # if in_data["frame_id"]!= "s000460" and  in_data["frame_id"]!= "r000460" and in_data["frame_id"]!= "s000474" and  in_data["frame_id"]!= "r000474" and in_data["frame_id"]!= "s1000460" and  in_data["frame_id"]!= "r1000460" and in_data["frame_id"]!= "s1000474" and  in_data["frame_id"]!= "r1000474" :
        #     continue
        if in_data["frame_id"][0][1:2]!="1":
            print("\n",in_data["frame_id"][0][1:2],"continue",in_data["frame_id"][0])
            continue
        # 条件文を簡潔に
        elif in_data["frame_id"][0][1:] not in invalid_ids:
            print(in_data["frame_id"][0][1:])
            continue
            
        # if in_data['frame_id'] == in_data_num:
        #     break 

    #入力データの形式を変更
        for key, val in in_data.items():
            if key == 'camera_imgs':
                in_data[key] = val.cuda()
            elif not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id','scene_name_info', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos','type_cam','target_gt_name']:
                continue
            elif key in ['images']:
                in_data[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                in_data[key] = torch.from_numpy(val).int().cuda()
            else:
                in_data[key] = torch.from_numpy(val).float().cuda()
        
        #データを入力
        # integ_dir = "./grad_base_DIVP_noaug2_16/pred_pc/"
        integ_dir = "./grad_base_DIVP/100m/pred_pc/"
        pred4_dir = args.save_file_name#"/pred_pcdV4"
        model_result_dir = "/" + args.Hmodel
        
        ###nuScenes用
        # save_dir = integ_dir + pred4_dir + model_result_dir +"/"+ in_data["scene_name_info"][0] + "/"+ in_data["frame_id"][0]
        ##########
        save_dir = integ_dir + pred4_dir + model_result_dir +"/"+ in_data["frame_id"][0]
        
        
        os.makedirs(save_dir, exist_ok=True)
        #ve_mode_1 = 'GuidedODAM'
        #mode_lst = ['ODAM', 'GradCAM', 'GuidedODAM', 'GuidedGradCAM', 'Guided']
        mode_lst = ['GradCAM']
        #visual_explanation(ve_mode_1, model, in_data, integ_dir, pred4_dir, save_mode=False, deletion=False, deletion_idx=None)
        if args.Hmodel == None:
            ve_mode_1 = 'ODAM'
        else:
            ve_mode_1 = args.Hmodel#,'GuidedODAM'#'ODAM'
        bp_scholar = args.bp_scholar
        grad_layer = args.grad_layer
        if grad_layer == None:
            grad_layer = 'spatial_features_2d_block'
        #grad_layer = ['spatial_features_2d_block', 'spatial_features_2d_deblock', 'spatial_features_2d']
        layer_number = args.layer_number
        # print(save_dir)
        # breakpoint()
        pcd_idx = visual_explanation(ve_mode_1, model, in_data, save_dir, grad_layer, layer_number, bp_scholar, save_mode=True, deletion=False, deletion_idx=None)
        gc.collect()
    #print('重要度の高いと思われる点群データを消去してモデルへ再入力')
    #重要度の高いと思われる点群データを消去してモデルへ再入力
    #pcd_idx = visual_explanation(ve_mode_1, model, in_data, integ_dir, pred4_dir, save_mode=False, deletion=True, deletion_idx=pcd_idx)
    #visual_explanation(in_data, deletion=False, deletion_idx=None)
def translate_boxes_to_open3d_instance(gt_boxes,mode=True):
    import open3d as o3d
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]

    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    vertices = np.asarray(line_set.points)
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    if mode == True:
        return line_set, box3d,vertices
    else:
        return vertices

def box_2d(box_3d,lidar2image):
    box_3d = box_3d.detach().numpy()
    lidar2image = lidar2image.detach().numpy()
    _,_,box_corner = translate_boxes_to_open3d_instance(box_3d)

    # lidar2camera_r = np.linalg.inv(input_dict["sensor2lidar_rotation"])
    # lidar2camera_t = (
    #     input_dict["sensor2lidar_translation"] @ lidar2camera_r.T
    # )
    # lidar2camera_rt = np.eye(4).astype(np.float32)
    # lidar2camera_rt[:3, :3] = lidar2camera_r.T
    # lidar2camera_rt[3, :3] = -lidar2camera_t
    # camera_intrinsics = np.eye(4).astype(np.float32)
    # camera_intrinsics[:3, :3] = input_dict["camera_intrinsics"]
    # lidar2image = camera_intrinsics @ lidar2camera_rt.T
   
    cur_coords = np.array(box_corner)
    cur_coords = lidar2image[:3, :3] @ cur_coords.T
    cur_coords += lidar2image[:3, 3].reshape((3, 1))

    # # 画像座標での座標値
    dists = np.clip(cur_coords[2], 1e-5, 1e5)
    x_coords = ((cur_coords[0] / dists ) * 0.48)-32
    y_coords = (((cur_coords[1] / dists )-70) * 0.48)-176


    
    x_min = int(min(x_coords))
    x_max = int(max(x_coords))
    y_min = int(min(y_coords))
    y_max = int(max(y_coords))
    
    # if x_max-704 <10 and x_max-704 >0:
    #     x_max -= 8
    # if y_max-256 < 10 and y_max-256 >0:
    #     y_max-=8
    # if 704-x_min >-10 and 704-x_min <0:
    #     x_min +=8
    # if 704-y_min >-10 and 704-y_min <0:
    #     y_min +=8
    return x_min,x_max,y_min,y_max


#視覚的説明を実行する．返り値として，attentionがかかる点群のtensor型のインデックスを返す
def visual_explanation(ve_mode_1, model, in_data2, save_dir, grad_layer, layer_number, bp_scholar=None, save_mode=False, deletion=False, deletion_idx=None):

    if deletion:
        copied_data = in_data2['voxels'].clone()
        copied_data[deletion_idx[0], deletion_idx[1], :] = copied_data[deletion_idx[0], deletion_idx[1], :] * 0
        in_data2['voxels'] = copied_data.clone()

    ################################データをモデルへ入力################################
    in_data2['voxels'] = in_data2['voxels'].requires_grad_()
    batch_dict, pred_dicts = model(in_data2) 
    
    # print(batch_dict.keys())
    # print(batch_dict["img_process_infos"])
    # breakpoint()
    
    # print("---------------------------------")
    # # print("key",batch_dict.keys())
    # print("points",batch_dict['points'].size())
    # print("voxel",batch_dict['voxels'].spatial_shape,batch_dict['voxel_coords'].size())
    # print("vox",batch_dict['voxels'][0])
    # print('encoded_spconv_tensor_feat',batch_dict['encoded_spconv_tensor'].features.shape)
    # print('encoded_spconv_tensor_ind',batch_dict['encoded_spconv_tensor'].indices.shape)
    # print("---------------------------------")
    # breakpoint()
    # input_sp_tensor.spatial_shape [54, 2000, 2000]
    input_sp_tensor_spatial_shape = [40, 2000, 2000] # 入力のグリッドサイズ(200/0.1, 200/0.2 , 8/0.2)
    f_spatial_shape = batch_dict['encoded_spconv_tensor'].spatial_shape
    
    Feat_Heat = torch.zeros(f_spatial_shape[0],f_spatial_shape[1],f_spatial_shape[2])

    # breakpoint()
    
    
    ####################Voxelの中身を確認用##########################
    # for i in range(Min, Max, 1):  # (開始, 終了+1, ステップ)
    #     print("---------------------------------")
    #     # print(i,":",batch_dict['voxel_coords'][batch_dict['voxel_coords'][:, 1] == i])
    #     print("---------------------------------")
    #     coo = copy.deepcopy(batch_dict['encoded_spconv_tensor'].indices[batch_dict['encoded_spconv_tensor'].indices[:, 1] == i])
    #     cooo = copy.deepcopy(batch_dict['encoded_spconv_tensor'].indices[batch_dict['encoded_spconv_tensor'].indices[:, 1] == i])
    #     kaburi = "False"
    #     for s_i,single in enumerate(coo):
    #         for ss_i,ssingle in enumerate(cooo):
    #             if s_i != ss_i:
    #                 if torch.equal(single, ssingle):
    #                     kaburi = "True"
    #     print(i,":",kaburi)


    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # ボクセル数、点群数、座標データ
    # # 例としてランダムデータを使用
    # # 実際にはこれらをデータとして与える
    # voxel_count = batch_dict['voxels'].size()[0]
    # voxels = batch_dict['voxels'][:,:,:3].to('cpu').detach().numpy().copy()
    # voxels_coord = batch_dict['voxel_coords'].to('cpu').detach().numpy().copy()

    # # ボクセルサイズ (0.1, 0.1, 0.15)
    # voxel_size = [0.1, 0.1, 0.15]

    # # 各ボクセルの座標データにボクセルサイズを適用
    # # x, y, zのそれぞれの座標値を拡大
    # # voxels_scaled = np.zeros_like(voxels)
    # # voxels_scaled[:, :, 0] = voxels[:, :, 0] * voxel_size[0]
    # # voxels_scaled[:, :, 1] = voxels[:, :, 1] * voxel_size[1]
    # # voxels_scaled[:, :, 2] = voxels[:, :, 2] * voxel_size[2]

    # # 可視化
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 点群の描画
    # for i in range(voxel_count):
    #     ax.scatter(voxels_coord[i, 1], voxels_coord[i, 2], voxels_coord[i,3], label=f'Voxel {i+1}')

    # # 軸ラベル
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # ax.view_init(elev=90, azim=-90)
    # plt.legend()
    # plt.show()
    # plt.savefig('voxel_visualization.png')
    # breakpoint()
    #############
    
    backprop_map_org = batch_dict['encoded_spconv_tensor'].features #torch.tensor
    num_voxel = batch_dict['encoded_spconv_tensor'].features.shape[0]

    if 'list' in str(type(backprop_map_org)):
        if layer_number == None:
            backprop_map = backprop_map_org
        else:
            backprop_map = [backprop_map_org[layer_number]]
    else:
        backprop_map = [backprop_map_org]

    #検出したクラススコア
    pred_scores = pred_dicts[0]['pred_scores']
    #検出したbbox
    pred_boxes = pred_dicts[0]['pred_boxes']
    #検出したラベル
    pred_labels = pred_dicts[0]['pred_labels']

    # box_loss = tb_dict['loc_loss_src'].squeeze(dim=0).sum(dim=-1)
    # cls_loss = tb_dict['cls_loss_src'].squeeze(dim=0).sum(dim=-1)

    score_threshold = {1:0.1, 2:0.1 ,3:0.1,4:0.1,5:0.1,6:0.1,7:0.1,8:0.1,9:0.1,10:0.1}
    eps = torch.finfo(torch.float32).eps

    heatmap_lst = []
    if bp_scholar == None:
        plop_lst = [0,1,2,3,4,5,6,7]
    else:
        plop_lst = [bp_scholar]
    #################################################################################
    
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # 逆正規化の処理
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    # camera_imgs を逆正規化
    in_data2["camera_imgs"] = [unnormalize(img) for img in in_data2["camera_imgs"]]
    #################################正規化の逆処理#####################################
    
    
    ans_pcd_lst = []
    #################################勾配計算の実行#####################################
    for pred_count in range(0, int(pred_labels.shape[0])):
        #閾値の決定
        thresh = score_threshold[int(pred_labels[pred_count])]
        point_mapsM = [] #Guidedの点群データに対するattentionを保存する

        point_mapM2 = []
        ans_heatmap_lstM2 = []

        #閾値以下の検出結果を除外
        if pred_scores[pred_count] >= thresh:
            #複数の畳み込み層に対してヒートマップを計算し，あとで加算する
            for nummm,num_cam_param in enumerate(in_data2["lidar2image"][0]):
                pred_solo = pred_boxes[pred_count].clone().cpu()
                param = num_cam_param.clone().cpu()
                x_min,x_max,y_min,y_max = box_2d(pred_solo,param)
                # print(x_min,x_max,y_min,y_max)
                pred_cam_scene = None
                if (((x_min>=0)&(x_min<=704))&((x_max>=0)&(x_max<=704))&((y_min>=0)&(y_min<=256))&((y_max>=0)&(y_max<=256))):
                    pred_cam_scene = copy.deepcopy(nummm)
                    break
            if pred_cam_scene is None:
                    continue

            heat_map_pcd = [] #guidedのattentionを保存
            for b_map in backprop_map:
                # heat_map3_pcd = torch.zeros(w*2, h*2, 32).to('cuda:0')
                for count_scholar, p_lst in enumerate(plop_lst):
                    #backpropさせる出力を決定
                    if p_lst == 7:#boxのスコア
                        focus_scholar = pred_scores[pred_count]
                    else:#boxのパラメータ
                        focus_scholar = pred_boxes[pred_count,p_lst]
                    
                    #勾配の計算

                    grad = torch.autograd.grad(focus_scholar, b_map, retain_graph=True)[0]
                    

                    ###############視覚的説明手法ごとの計算を実行##################
                    if ve_mode_1 == 'ODAM' or ve_mode_1 == 'GuidedODAM':
                        heat_map = F.relu_((grad * b_map.detach()).sum(1))
                        # print('heatmap1 : ', heat_map.shape,heat_map)
                        
                        for sub_heat,heat_coords in zip(heat_map,batch_dict['encoded_spconv_tensor'].indices):
                            
                            pp_heat = sub_heat.clone().to('cpu').detach()
                            xyz = heat_coords.clone().to('cpu').detach()
                            Feat_Heat[xyz[1]][xyz[2]][xyz[3]] = pp_heat
                        # print(Feat_Heat.size())
                        output_data = F.interpolate(Feat_Heat.unsqueeze(0).unsqueeze(0), 
                                                    size=(input_sp_tensor_spatial_shape[0],
                                                            input_sp_tensor_spatial_shape[1],
                                                            input_sp_tensor_spatial_shape[2]), 
                                                    mode='trilinear', align_corners=False)
                        # print(output_data.size())
                        # heat_map = interpolation(heat_map)
                        #heat_map = tf(heat_map)
                        # print('heatmap2 : ', output_data.shape)
                        heat_map = output_data.squeeze(0).squeeze(0)
                        # break
                    else:
                        pass

                    if ve_mode_1 == 'GradCAM'or ve_mode_1 == 'GuidedGradCAM':
                        alpha = grad.clone().view(-1, int(b_map.shape[2])*int(b_map.shape[3]))
                        alpha = torch.mean(alpha, axis=1)
                        heat_map = F.relu_((b_map.detach().squeeze(0) * alpha.view(-1,1,1)).sum(0)).unsqueeze(0)
                        # print('heatmap : ',heat_map.shape, alpha.shape)
                        breakpoint(9)
                        heat_map = interpolation(heat_map)
                        heat_map = heat_map.squeeze(0)
                    else:
                        pass

                    if ve_mode_1 == 'Guided'or ve_mode_1 == 'GuidedODAM'or ve_mode_1 == 'GuidedGradCAM':
                        relu1 = torch.nn.ReLU()
                        relu2 = torch.nn.ReLU()

                        focusmap2 = voxels
                        focusmap2_points *= 0
                        focusmap2_gcams *= 0

                        sf_idx = torch.where((focusmap2 <= 0))
                        grad_relu = torch.autograd.grad(focus_scholar, focusmap2, retain_graph=True)[0]
                        grad_relu[sf_idx] *= 0
                        grad_relu = relu2(grad_relu)

                        focusmap2_relu = relu1(focusmap2.detach())
                        focusmap2_V2 = F.relu_((grad_relu * focusmap2_relu.detach()))

                        """点群データへのattention"""
                        focusmap2_points[coords[:,2].reshape(-1), coords[:,3].reshape(-1)] += focusmap2_V2#.view(pillar_num, -1, 4)

                        if ve_mode_1 == 'GuidedGradCAM':
                            heat_map2 = torch.max(focusmap2_points.clone(), dim=-1)[0].clone()
                            heat_map2 = torch.max(heat_map2, dim=-1)[0]
                            heat_map_b = heat_map.clone()
                            heat_map = heat_map * heat_map2 #ヒートマップの作成

                            heat_map32_pcd = (torch.max(focusmap2_points.clone(), dim=-1)[0].clone() * heat_map_b.clone().view(w*2, h*2, -1))               
                            heat_map32_pcd = (heat_map32_pcd - heat_map32_pcd.min()) / (heat_map32_pcd.max() - heat_map32_pcd.min())
                            point_mapM2.append(heat_map32_pcd.clone())
                        elif ve_mode_1 == 'GuidedODAM':
                            heat_map2 = torch.max(focusmap2_points.clone(), dim=-1)[0].clone()
                            heat_map2 = torch.max(heat_map2, dim=-1)[0]
                            heat_map_b = heat_map.clone()
                            heat_map = heat_map * heat_map2 #ヒートマップの作成

                            if heat_map_b.max() - heat_map_b.min() > 0:
                                heat_map_b = (heat_map_b - heat_map_b.min()) / (heat_map_b.max() - heat_map_b.min())

                            if focusmap2_points.max() - focusmap2_points.min() > 0:
                                focusmap2_points = (focusmap2_points - focusmap2_points.min()) / (focusmap2_points.max() - focusmap2_points.min())

                            heat_map32_pcd = (torch.max(focusmap2_points.clone(), dim=-1)[0].clone() + heat_map_b.clone().view(w*2, h*2, -1))               
                            heat_map32_pcd = heat_map32_pcd * heat_map32_pcd
                            heat_map32_pcd = (heat_map32_pcd - heat_map32_pcd.min()) / (heat_map32_pcd.max() - heat_map32_pcd.min())
                            point_mapM2.append(heat_map32_pcd.clone())

                        else: #Guided
                            heat_map = torch.max(focusmap2_points.clone(), dim=-1)[0].clone()
                            heat_map = torch.max(heat_map, dim=-1)[0]
                            heat_map32_pcd = torch.max(focusmap2_points.clone(), dim=-1)[0].clone()
                            if heat_map32_pcd.max() - heat_map32_pcd.min() > 0:
                                heat_map32_pcd = (heat_map32_pcd - heat_map32_pcd.min()) / (heat_map32_pcd.max() - heat_map32_pcd.min())
                            point_mapM2.append(heat_map32_pcd.clone())
                            # print(torch.max(heat_map32_pcd), torch.min(heat_map32_pcd))

                    else:
                        pass
                    ###############################################################
                    #ヒートマップをリストへ保存
                    heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min()).clamp(min=eps)
                    ans_heatmap_lstM2.append(heat_map)
                    # break
                    """
                    リストに保存したヒートマップを加算し点群データへ頒布する
                    guided : 出力された点群データへ対する重みを加算して保存
                    
                    """
            ############################畳み込み層ごとのヒートマップ計算はここまで###############################

            ############################ヒートマップを合算###############################  

            # print("def",torch.stack(copy.deepcopy(ans_heatmap_lstM2)).size())
            # breakpoint()
            
            ####Cat#####
            ans_heatmap = torch.max(torch.stack(copy.deepcopy(ans_heatmap_lstM2), dim=-1), dim=-1)[0]
            ans_heatmap_lstM2.append(ans_heatmap)
            
            
            
            ########################## to Poincloud N*5 ################################
            all_pcd_plus_color = []
            ans_voxel_copy = batch_dict['voxels'].clone()
            for iiii,lst_M2 in enumerate(ans_heatmap_lstM2):
                pcd_plus_color = []
                if 'Guided' not in ve_mode_1:
                    for id_voxel, ans_voxel in enumerate(ans_voxel_copy):
                        ans_coord = batch_dict['voxel_coords'][id_voxel].clone().to('cpu').detach()
                        # print(ans_coord,lst_M2.size(),ans_voxel.size(),batch_dict['voxel_coords'].size())
                        z = int(ans_coord[1])
                        y = int(ans_coord[2])
                        x = int(ans_coord[3])
                        pcd_color = lst_M2[z][y][x].clone().to('cpu').detach()
                        for one_point in ans_voxel:
                            list_one_point = one_point.clone().tolist()
                            list_one_point.append(pcd_color)
                            pcd_plus_color.append(list_one_point)
                cccc = np.array(pcd_plus_color)
                # print(cccc.shape)
                # np.save("./demo_pc_vis/demo_"+str(iiii)+"_pc.npy",cccc)
                all_pcd_plus_color.append(cccc)
            
            project_pc = []
            for ans_prot in all_pcd_plus_color:
                each_prot = []
                color = copy.deepcopy(ans_prot[:,4])
                for nummm,num_cam_param in enumerate(in_data2["lidar2image"][0]):
                    point = copy.deepcopy(ans_prot[:,:3])
                    param = num_cam_param.clone().cpu()
                    prot_x,prot_y = project(point,param)
                    prot_xy = np.hstack((prot_x.reshape(-1, 1),prot_y.reshape(-1, 1)))
                    each_prot.append(copy.deepcopy(prot_xy))
                project_pc.append(copy.deepcopy(each_prot))
            ##########################################################################            
            
            
            import cv2
            from PIL import Image
            # import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            cloclom_label = {"0":"x","1":"y","2":"z","3":"l","4":"h","5":"w","6":"rot","7":"score","8":"cat"}
            # グラフのレイアウトを調整（ラベル列 + 画像列）
            fig = plt.figure(figsize=(15, 8))
            gs = gridspec.GridSpec(9, 7, width_ratios=[0.5, 1, 1, 1, 1, 1, 1])  # 最初の列をラベル用に

            for row_num,odam_maps in enumerate(all_pcd_plus_color):
                # ラベルを左隅に配置
                ax_label = fig.add_subplot(gs[row_num, 0])
                ax_label.axis('off')  # 軸を非表示に
                ax_label.text(0.5, 0.5, cloclom_label[str(row_num)], va='center', ha='center', fontsize=12, rotation=90, transform=ax_label.transAxes)
                
                # 画像を右の6列に表示
                for c_c_num in range(1):
                    att = odam_maps[:,4] * 255.
                    # print(att.shape)
                    img_array = np.clip(in_data2["camera_imgs"][0][c_c_num].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8).copy()
                    
                    ax = fig.add_subplot(gs[row_num, c_c_num + 1])  # 画像を追加
                    ax.axis('off')
                    
                    color = cv2.applyColorMap(att.astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].reshape(att.shape[0],3) # cv2 to plt
                    c_ret = np.clip(img_array[:, :, ::-1], 0,255).astype(np.uint8)
                    
                    
                    
                    project_pc_np = np.array(project_pc[row_num][c_c_num])
                    project_color_np = np.array(color)
                    # 条件に基づいてフィルタリング
                    filtered_pc = project_pc_np[
                        ((project_pc_np[:, 0] > 0) & (project_pc_np[:, 0] < 704)) &
                        ((project_pc_np[:, 1] > 0) & (project_pc_np[:, 1] < 256))
                    ]
                    filtered_color = project_color_np[
                        ((project_pc_np[:, 0] > 0) & (project_pc_np[:, 0] < 704)) &
                        ((project_pc_np[:, 1] > 0) & (project_pc_np[:, 1] < 256))]
                    
                    # print(filtered_pc.shape,filtered_color.shape)
                    # breakpoint()
                    
                    for f_c,f_pc in zip(filtered_color,filtered_pc):
                    #     if (int(cv_prot[0])<704 and int(cv_prot[0])>0 ) and (int(cv_prot[1])<256 and int(cv_prot[1])>0 ):
                    #         # print(cv_prot,color.shape)
                    #         # breakpoint()
                        cv2.circle(c_ret,(int(f_pc[0]),int(f_pc[1])),2,(int(f_c[0]),int(f_c[1]),int(f_c[2])),-1)
                    
                    # 描画する点の座標と色を格納するリストを用意

                    if c_c_num == pred_cam_scene:
                        # print(f"img_array min value: {img_array.min()}, max value: {img_array.max()}")
                        # print(f"Shape: {img_array.shape}, Dtype: {img_array.dtype}")
                        cv2.rectangle(c_ret, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                    
                    ax.imshow(c_ret)

            fig.tight_layout()
            fig.savefig(save_dir + "/" + str(pred_count)+"_"+str(pred_scores[pred_count]) + ".png")
            plt.close(fig)
            # breakpoint()

def return_points_weight(voxels_w, ans_heatmap, coords):
    #for j in range(0,int(coords.shape[0])):
    #    voxels_w[coords[j,2], coords[j,3]] += ans_heatmap[coords[j,2], coords[j,3]]
    voxels_w += ans_heatmap

    #ヒートマップの画素値をvoxel内の点群へ頒布 2
    pcd_color = point_colors(voxels_w.clone())
    return pcd_color, voxels_w

#ヒートマップの作成
def toHeatmap(img):
    x = int(img.shape[0])
    y = int(img.shape[1])
    img = (img*255).reshape(-1)
    cm = plt.get_cmap('jet')
    img = np.array([cm(int(np.round(xi)))[:3] for xi in img])
    img = img.reshape(x, y, 3)
    return np.flipud(img)

def modify_map(ans_heatmap, eps):
    ans_heatmap = (ans_heatmap - ans_heatmap.min()) / (ans_heatmap.max() - ans_heatmap.min()).clamp(min=eps)
    heat_map_np = toHeatmap(ans_heatmap.to('cpu').detach().numpy().copy()) * 255

    heat_map_np2 = []
    for i in range(int(heat_map_np.shape[-1])-1,-1,-1):
        heat_map_np2.append(heat_map_np[:,:,i])
    heat_map_np2 = np.stack(heat_map_np2, axis=-1)
    return heat_map_np2

def point_colors(point_w): #点群データに色を設定
    point_r = point_w.clone() * 0
    point_g = point_w.clone() * 0
    point_b = point_w.clone() * 0

    #phase 1 0   ~ 1/4
    R = [0, 1/4]
    p_idx = torch.where((point_w >= R[0]) & (point_w < R[1]))
    point_g[p_idx] = point_w[p_idx] * 4
    point_b[p_idx] = 1

    #phase 2 1/4 ~ 1/2
    R = [1/4, 2/4]
    p_idx = torch.where((point_w >= R[0]) & (point_w < R[1]))
    point_g[p_idx] = 1
    point_b[p_idx] = (point_w[p_idx] * (-4)) + 2

    #phase 3 1/2 ~ 3/4
    R = [2/4, 3/4]
    p_idx = torch.where((point_w >= R[0]) & (point_w < R[1]))
    point_r[p_idx] = (point_w[p_idx] * 4) - 2
    point_g[p_idx] = 1

    #phase 4 3/4 ~ 1
    R = [3/4, 4/4]
    p_idx = torch.where((point_w >= R[0]) & (point_w <= R[1]))
    point_r[p_idx] = 1
    point_g[p_idx] = (point_w[p_idx] * (-4)) + 4

    ans = torch.stack([point_r, point_g, point_b],dim=-1)
    return ans

if __name__ == '__main__':
    main()
