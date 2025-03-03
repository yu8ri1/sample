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
import torchvision.transforms as transforms
import torchvision
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

import numpy as np
import torch.nn.functional as F

import cv2
from matplotlib import pyplot as plt

import copy
from tqdm import tqdm
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
    parser.add_argument('--save_file_name', type=str, default=None, help='selsect grad_layer')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


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

    # train_set, train_loader, train_sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers,
    #     logger=logger,
    #     training=True,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    #     total_epochs=args.epochs,
    #     seed=666 if args.fix_random_seed else None
    # )

    # #モデルの定義
    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # #print(model)
    # if args.sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.cuda()

    #optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    #学習済みモデルの読み込み
    # chk_pt_kitti = torch.load(args.ckpt)

    # #学習済みの重みをモデルへ移行
    # model.load_state_dict(chk_pt_kitti['model_state'])

    #データローダを定義
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

    ##全
    # for i in range(len(in_iter)):
    for i in tqdm(range(len(in_iter)), desc="CAM_Processing", total=len(in_iter), ncols=100):
        in_data = next(in_iter)
        # print(in_data["img_process_infos"])
        # if int(in_data["frame_id"][0][1:])<420:
        #     continue
        ##nuScenes用
        # print(in_data.keys(),in_data["scene_name_info"])
        # breakpoint()
        # if in_data["scene_name_info"] != "scene-0638" and in_data["scene_name_info"] != "scene-1066"  and in_data["scene_name_info"] != "scene-0520" and in_data["scene_name_info"] != "scene-0108":
        #     continue
        # if in_data["frame_id"]!= "s000460" and  in_data["frame_id"]!= "r000460" and in_data["frame_id"]!= "s000474" and  in_data["frame_id"]!= "r000474" and in_data["frame_id"]!= "s1000460" and  in_data["frame_id"]!= "r1000460" and in_data["frame_id"]!= "s1000474" and  in_data["frame_id"]!= "r1000474" :
        #     continue
        # print("\n",in_data["frame_id"][0][1:2],in_data["frame_id"][0])
        # breakpoint()
        invalid_ids = [
            "1000410", "1000420", "1000430", "1000440", "1000450","1000460","1000469", "1000474",
            "1000885", "1000895", "1000905", "1000915", "1000925", "1000935", "1000945",
            "1000955", "1001370", "1001380", "1001390", "1001400", "1001410", "1001420",
            "1001430", "1001440"
        ]
        if in_data["frame_id"][0][1:2]!="1":
            print("\n",in_data["frame_id"][0][1:2],"continue",in_data["frame_id"][0])
            continue
        elif in_data["frame_id"][0][1:] not in invalid_ids:
            print(in_data["frame_id"][0][1:])
            continue
        # breakpoint()
        #######
        
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


        integ_dir = "./grad_base_DIVP/200m/pred_cam/"
        pred4_dir = args.save_file_name#"/pred_pcdV4"
        model_result_dir = "/" + args.Hmodel
        
        ###nuScenes用
        # save_dir = integ_dir + pred4_dir + model_result_dir +"/"+ in_data["scene_name_info"][0] + "/"+ in_data["frame_id"][0]
        ##########
        save_dir = integ_dir + pred4_dir + model_result_dir +"/"+ in_data["frame_id"][0]
        
        os.makedirs(save_dir, exist_ok=True)
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

#視覚的説明を実行する．返り値として，attentionがかかる点群のtensor型のインデックスを返す
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
    x_coords = ((cur_coords[0] / dists ) * 0.48) -32
    y_coords = (((cur_coords[1] / dists )-70) * 0.48) -176
    # dists = np.clip(cur_coords[2], 1e-5, 1e5)
    # x_coords = ((cur_coords[0] / dists ) * 0.48)-32
    # y_coords = (((cur_coords[1] / dists )-70) * 0.48)-17
    
    # dists = np.clip(cur_coords[2], 1e-5, 1e5)
    # x_coords = ((cur_coords[0] / dists ))
    # y_coords = (((cur_coords[1] / dists )-70))

    
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



def visual_explanation(ve_mode_1, model, in_data2, save_dir, grad_layer, layer_number, bp_scholar=None, save_mode=False, deletion=False, deletion_idx=None):
    # if deletion:
    #     copied_data = in_data2['camera_imgs'].clone()
    #     copied_data[deletion_idx[0], deletion_idx[1], :] = copied_data[deletion_idx[0], deletion_idx[1], :] * 0
    #     in_data2['voxels'] = copied_data.clone()

    ################################データをモデルへ入力################################

    in_data2['camera_imgs'] = in_data2['camera_imgs'].requires_grad_()
    in_data2['voxels'] = in_data2['voxels'].requires_grad_()
    batch_dict, pred_dicts = model(in_data2) #推論
    # print(pred_dicts)
    # breakpoint()
    #################################################################################
    ##########################勾配計算に使用する変数を定義################################
    # backprop_map_org = batch_dict['multi_scale_3d_features']["x_conv3"]#([15069, 16])([21935, 32])([14600, 64])
    backprop_map_org = batch_dict['image_features']
    w = int(in_data2['camera_imgs'][0][0].size(1))
    h = int(in_data2['camera_imgs'][0][0].size(2))
    # print(w,h)
    
    interpolation = torchvision.transforms.Resize((w,h))

    # # focusmap2_points = torch.zeros(w*2,h*2,32,4).to('cuda:0')
    # # focusmap2_gcams = torch.zeros(w*2,h*2,32,4)
    # voxels_w = torch.zeros(w*2, h*2, 32)
    # focusmap2_points = torch.zeros(w*2,h*2,10,4).to('cuda:0')
    # focusmap2_gcams = torch.zeros(w*2,h*2,10,4)
    voxels_w = torch.zeros(w,h,3)

    # #勾配を確認する層の定義
    # backprop_map_org = batch_dict[grad_layer]

    backprop_map = []
    if 'list' in str(type(backprop_map_org)):
        if layer_number == None:
            backprop_map = backprop_map_org
        else:
            backprop_map = [backprop_map_org[layer_number]]
    else:
        backprop_map = [backprop_map_org]

    # print(batch_dict["img_process_infos"])
    voxels = batch_dict['voxels'] #(pillar_num, 32, 4)
    coords = batch_dict['voxel_coords'].to('cpu').numpy().copy()
    coords = np.array(coords, dtype=np.int64) #(Pillrの数, 4) (:, 2:)に画素の位置が記録
    # print(batch_dict['voxel_coords'].size())
    # breakpoint()
    pillar_num = int(coords.shape[0])
    #生きている点群データのピックアップ
    mpw = voxels.detach().view(-1, 4)
    mpw_idx = torch.where((mpw[:,0] != 0) & (mpw[:,1] != 0) & (mpw[:,2] != 0))
    mpw_idx2 = torch.where((voxels[:,:,0] != 0) & (voxels[:,:,1] != 0) & (voxels[:,:,2] != 0))
    point_clouds = mpw[mpw_idx].to('cpu')
    point_clouds = point_clouds.detach().numpy().copy()

    #検出したクラススコア
    pred_scores = pred_dicts[0]['pred_scores']
    #検出したbbox
    pred_boxes = pred_dicts[0]['pred_boxes']
    #検出したラベル
    pred_labels = pred_dicts[0]['pred_labels']
    
    # print(pred_row.size(),pred_scores.size())
    # breakpoint()
    
    #boxとクラスのloss
    # selected_idx = pred_dicts[0]['selected_idx'].to('cpu')
    # box_loss = tb_dict['loc_loss_src'].squeeze(dim=0)[selected_idx].sum(dim=-1)
    # cls_loss = tb_dict['cls_loss_src'].squeeze(dim=0)[selected_idx].sum(dim=-1)
    # box_loss = tb_dict['loc_loss_src'].squeeze(dim=0).sum(dim=-1)
    # cls_loss = tb_dict['cls_loss_src'].squeeze(dim=0).sum(dim=-1)

    score_threshold = {1:0.1, 2:0.1 ,3:0.1,4:0.1,5:0.1,6:0.1,7:0.1,8:0.1,9:0.1,10:0.1}
    # score_threshold = {1:0, 2:0 ,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
    eps = torch.finfo(torch.float32).eps

    heatmap_lst = []
    if bp_scholar == None:
        plop_lst = [0,1,2,3,4,5,6,7]
    else:
        plop_lst = [bp_scholar]
        
    #################################################################################
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # mean = torch.tensor([123.675, 116.28, 103.53])
    # std = torch.tensor([58.395, 57.12, 57.375])
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
        # ans_heatmap_lstM2 = []
        ans_heatmap_lstM2 = {"0":[],"1":[],"2":[],"3":[],"4":[],"5":[],"6":[],"7":[]}

        #閾値以下の検出結果を除外
        if pred_scores[pred_count] >= thresh:
            
            #複数の畳み込み層に対してヒートマップを計算し，あとで加算する
            # print(in_data2.keys())
            for nummm,num_cam_param in enumerate(in_data2["lidar2image"][0]):
                pred_solo = pred_boxes[pred_count].clone().cpu()
                param = num_cam_param.clone().cpu()
                x_min,x_max,y_min,y_max = box_2d(pred_solo,param)
                # print(x_min,x_max,y_min,y_max)
                pred_cam_scene = None
                if (((x_min>=0)&(x_min<=704))&((x_max>=0)&(x_max<=704))&((y_min>=0)&(y_min<=256))&((y_max>=0)&(y_max<=256))):
                    pred_cam_scene = copy.deepcopy(nummm)
                    break
                # if (((x_min>=0)&(x_min<=800))&((x_max>=0)&(x_max<=800))&((y_min>=0)&(y_min<=450))&((y_max>=0)&(y_max<=450))):
                #     pred_cam_scene = copy.deepcopy(nummm)
                #     break
                # if (((x_min>=0)&(x_min<=1600))&((x_max>=0)&(x_max<=1600))&((y_min>=0)&(y_min<=900))&((y_max>=0)&(y_max<=900))):
                #     pred_cam_scene = copy.deepcopy(nummm)
                #     break
            if pred_cam_scene is None:
                    continue
                # breakpoint()
            # print(x_min,x_max,y_min,y_max,pred_cam_scene)
            # breakpoint()
            
            for b_map in backprop_map:
                all_map_list = []
                # print(b_map.shape)
                for  num_cam,b_map_split in enumerate(b_map):
                    each_param_list = []
                    # grad_split_u = grad_split.unsqueeze(0) ## CHW--> BCHW
                    b_map_split_u = b_map_split.unsqueeze(0) ## CHW--> BCHW 
                    for count_scholar, p_lst in enumerate(plop_lst):
                        #backpropさせる出力を決定
                        if p_lst == 7:#boxのスコア
                            focus_scholar = pred_scores[pred_count]
                        else:#boxのパラメータ
                            focus_scholar = pred_boxes[pred_count,p_lst]

                        # torch.autograd.set_detect_anomaly(True)

                        grad = torch.autograd.grad(focus_scholar, backprop_map_org, retain_graph=True)[0]

                        grad_split_u = grad[num_cam].unsqueeze(0)
                        # print(b_map_split_u.size(),grad_split_u.size())
                        # breakpoint()
                        ###############視覚的説明手法ごとの計算を実行##################
                        if ve_mode_1 == 'ODAM' or ve_mode_1 == 'GuidedODAM':
                            heat_map = F.relu_((grad_split_u * b_map_split_u.detach()).sum(1))
                            heat_map = interpolation(heat_map)
                            heat_map = heat_map.squeeze(0)
                            ans_heatmap_lstM2[str(count_scholar)].append(copy.deepcopy(heat_map))
                            
                        else:
                            pass

                        if ve_mode_1 == 'GradCAM'or ve_mode_1 == 'GuidedGradCAM':
                            alpha = grad_split_u.clone().view(-1, int(b_map_split_u.shape[2])*int(b_map_split_u.shape[3])) # BCHW --> [B*C,H*W]
                            # alpha = grad.clone().view(-1, int(b_map_split_u.shape[2])*int(b_map_split_u.shape[3]))
                            alpha = torch.mean(alpha, axis=1)
                            # heat_map = F.relu_((b_map_split_u.detach().squeeze(0) * alpha.view(-1,1,1)).sum(0)).unsqueeze(0)
                            heat_map = F.relu(torch.sum(b_map_split_u.detach().squeeze(0)*alpha.view(-1,1,1),0)).unsqueeze(0)
                            heat_map = interpolation(heat_map)
                            # breakpoint()
                            heat_map = heat_map.squeeze(0)
                            # heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min()).clamp(min=eps)
                            ans_heatmap_lstM2[str(count_scholar)].append(copy.deepcopy(heat_map))

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
                
                for ahl_key, ahl in ans_heatmap_lstM2.items():
                    list2torch = torch.stack(copy.deepcopy(ahl))
                    list2torch = (list2torch - list2torch.min()) / (list2torch.max() - list2torch.min()).clamp(min=eps)
                    ans_heatmap_lstM2[ahl_key] = list2torch.cpu()
                
                # print(ans_heatmap_lstM2.keys())
                ans_heatmap_lstM2["cat"]=[]
                # print(ans_heatmap_lstM2["0"])
                # breakpoint()
                for num_f in range(1):
                    sub_list = []
                    for ahl_key, ahl in ans_heatmap_lstM2.items():
                        
                        if ahl_key != "cat" and ahl_key != "7":
                            sub_list.append(copy.deepcopy(ahl[num_f]))
                    sub_list_stack_max= torch.max(torch.stack(sub_list, dim=-1), dim=-1)[0]
                    ans_heatmap_lstM2["cat"].append(copy.deepcopy(sub_list_stack_max))
            ############################ヒートマップを合算###############################  
            # ans_heatmap = torch.max(torch.stack(ans_heatmap_lstM2, dim=-1), dim=-1)[0]
                import cv2
                from PIL import Image
                # import numpy as np
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                
                    
                cloclom_label = {"0":"x","1":"y","2":"z","3":"l","4":"h","5":"w","6":"rot","7":"score","cat":"concat"}
                # グラフのレイアウトを調整（ラベル列 + 画像列）
                fig = plt.figure(figsize=(15, 8))
                gs = gridspec.GridSpec(9, 7, width_ratios=[0.5, 1, 1, 1, 1, 1, 1])  # 最初の列をラベル用に

                for row_num, (key, odam_maps) in enumerate(ans_heatmap_lstM2.items()):
                    # ラベルを左隅に配置
                    
                    ax_label = fig.add_subplot(gs[row_num, 0])
                    ax_label.axis('off')  # 軸を非表示に
                    ax_label.text(0.5, 0.5, cloclom_label[str(key)], va='center', ha='center', fontsize=12, rotation=90, transform=ax_label.transAxes)
                    
                    # 画像を右の6列に表示
                    for c_c_num in range(1):
                        att = odam_maps[c_c_num].numpy() * 255.
                        
                        img_array = np.clip(in_data2["camera_imgs"][0][c_c_num].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8).copy()
                        
                        ax = fig.add_subplot(gs[row_num, c_c_num + 1])  # 画像を追加
                        ax.axis('off')
                        
                        color = cv2.applyColorMap(att.astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]  # cv2 to plt
                        c_ret = np.clip(img_array[:, :, ::-1] * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
                        
                        if c_c_num == pred_cam_scene:
                            cv2.rectangle(c_ret, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                        
                        ax.imshow(c_ret)

                fig.tight_layout()
                fig.savefig(save_dir + "/" + str(pred_count)+"_"+str(pred_scores[pred_count]) + ".png")
                plt.close(fig)


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
