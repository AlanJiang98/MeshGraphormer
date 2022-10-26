import os
from tkinter import image_names
os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import smplx
# import trimesh
# import pyrender
from smplx import MANO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.io.ply_io import load_ply, save_ply
from pytorch3d.utils import cameras_from_opencv_projection
import json
from pytorch3d.io import IO
import shutil

def get_synthetic_affine_matrix():
        # interhand image settings
    img_ori_height, img_ori_width = 512, 334
    davis_width, davis_height = 346, 260
    # get the affine matrix from RGB frame to event frame, and this affine matrix will influence the camera intrinsics
    src_points = np.float32([[0, 0], [img_ori_width / 2 - 1, 0], [img_ori_width / 2 - 1, img_ori_height]])
    dst_points = np.float32([[davis_width / 2 - img_ori_width / 2 * davis_height / img_ori_height, 0],
                             [davis_width / 2 - 1, 0],
                             [davis_width / 2 - 1, davis_height - 1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    K_affine = np.concatenate([affine_matrix, np.array([[0, 0, 1.0]])], axis=0)
    return K_affine

if __name__ == "__main__":
    data_dir = "/userhome/wangbingxuan/data/Interhand"
    if not os.path.exists(os.path.join(data_dir, "mask")):
        os.mkdir(os.path.join(data_dir, "mask"))
    mask_dir = os.path.join(data_dir, "mask")
    train_dir = os.path.join(data_dir, "train")
    event = True
    if event:
        K_affine = get_synthetic_affine_matrix()
        event_mask_dir = os.path.join(data_dir, "synthetic", "mask")
        if not os.path.exists(event_mask_dir):
            os.mkdir(event_mask_dir)
        mask_dir = event_mask_dir
    camera_keys = ['400002', '400004','400009','400018','400026', 
                    '400039','400041', '400053']
    annot_dict = {
        "cam_param": json.load(open(os.path.join(data_dir,'annot/InterHand2.6M_train_camera.json'),'r')),
        # "joint_param": json.load(open(os.path.join(data_dir,'annot/InterHand2.6M_train_joint_3d.json'),'r')),
        "mano_param": json.load(open(os.path.join(data_dir,'annot/InterHand2.6M_train_MANO_NeuralAnnot_new.json'),'r')),
    }
    device = torch.device('cuda:7')
    # K = {cam_id : torch.tensor(np.array([[annot_dict["cam_param"]['focal'][cam_id][0], 0, annot_dict["cam_param"]['princpt'][cam_id][0]],[0, annot_dict["cam_param"]['focal'][cam_id][1], annot_dict["cam_param"]['princpt'][cam_id][1]],[0, 0, 1.]]), dtype=torch.float32).view(3, 3) for cam_id in camera_keys}
    # R = {cam_id : torch.tensor(annot_dict["cam_param"]['camrot'][cam_id], dtype=torch.float32).view(3, 3) for cam_id in camera_keys}
    # t = {cam_id : torch.tensor(annot_dict["cam_param"]['campos'][cam_id], dtype=torch.float32).view(3, 3) for cam_id in camera_keys}
    
    
    hand_type = "right"
    smplx_path = "/userhome/wangbingxuan/code/data/model"
    mano_layer = {'right': MANO(smplx_path, ir_rhand=True, use_pca=False),
              'left': MANO(smplx_path, ir_rhand=False, use_pca=False)}
    if event:
        height, width = 260, 346
    else:
        image = cv2.imread(f"/userhome/wangbingxuan/code/data/Interhand/train/Capture0/0000_neutral_relaxed/cam400002/image0563.jpg")
        height, width = image.shape[0], image.shape[1]
    print(f"height: {height}, width: {width}")
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
            # print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1
    mano_layer[hand_type].to(device)
    print("start!")

    for seq in os.listdir(train_dir):
        cap_id =seq[-1]
        if event:
            K = {cam_id : torch.tensor(np.dot(K_affine,np.array([[annot_dict["cam_param"][cap_id]['focal'][cam_id][0], 0, annot_dict["cam_param"][cap_id]['princpt'][cam_id][0]],
                                                    [0, annot_dict["cam_param"][cap_id]['focal'][cam_id][1], annot_dict["cam_param"][cap_id]['princpt'][cam_id][1]],
                                                    [0, 0, 1.]])), dtype=torch.float32).view(3, 3) for cam_id in camera_keys}
        

        else:
            K = {cam_id : torch.tensor(np.array([[annot_dict["cam_param"][cap_id]['focal'][cam_id][0], 0, annot_dict["cam_param"][cap_id]['princpt'][cam_id][0]],
                                                    [0, annot_dict["cam_param"][cap_id]['focal'][cam_id][1], annot_dict["cam_param"][cap_id]['princpt'][cam_id][1]],
                                                    [0, 0, 1.]]), dtype=torch.float32).view(3, 3) for cam_id in camera_keys}
        R = {cam_id : torch.tensor(annot_dict["cam_param"][cap_id]['camrot'][cam_id], dtype=torch.float32).view(3, 3) for cam_id in camera_keys}
        t = {cam_id : torch.tensor(annot_dict["cam_param"][cap_id]['campos'][cam_id], dtype=torch.float32).view(3, 1) / 1000. for cam_id in camera_keys}
        # remember to change t as t = (-R @ t).view(3)
    # # for seq in["16"]:
        if not os.path.isdir(os.path.join(train_dir, seq)):
            continue
        if not os.path.exists(os.path.join(mask_dir, seq)):
            os.mkdir(os.path.join(mask_dir, seq))
        print(seq)
        for gesture in os.listdir(os.path.join(train_dir, seq)):
            manos = {'mano_trans':[],
                    'mano_rot_pose':[],
                    'mano_shape':[],
                    'mano_hand_pose':[],
            }
            image_names = []
            for idx_cam in range(len(camera_keys)):
                if not os.path.exists(os.path.join(train_dir, seq, gesture, "cam"+camera_keys[idx_cam])):
                    continue
                for image_name in os.listdir(os.path.join(train_dir, seq, gesture,"cam"+camera_keys[idx_cam])):
                    image_id = str(int(image_name[5:-4]))
                    if annot_dict["mano_param"][cap_id][image_id][hand_type]==None:
                        continue
                    image_names.append(image_name)
                    # print(cap_id, image_id)
                    manos["mano_trans"].append(annot_dict["mano_param"][cap_id][image_id][hand_type]["trans"])
                    manos["mano_rot_pose"].append(annot_dict["mano_param"][cap_id][image_id][hand_type]["pose"][:3])
                    manos["mano_shape"].append(annot_dict["mano_param"][cap_id][image_id][hand_type]["shape"])
                    manos["mano_hand_pose"].append(annot_dict["mano_param"][cap_id][image_id][hand_type]["pose"][3:])
                break
            if len(image_names)==0:
                continue
            split_mano = {'mano_trans':torch.tensor(manos["mano_trans"], device=device),
                        'mano_rot_pose':torch.tensor(manos["mano_rot_pose"], device=device),
                        'mano_shape':torch.tensor(manos["mano_shape"], device=device),
                        'mano_hand_pose':torch.tensor(manos["mano_hand_pose"], device=device),
                }
            output = mano_layer[hand_type](
                global_orient=split_mano['mano_rot_pose'].reshape(-1, 3),
                hand_pose=split_mano['mano_hand_pose'].reshape(-1, 45),
                betas=split_mano['mano_shape'].reshape(-1, 10),
                transl=split_mano['mano_trans'].reshape(-1, 3)
            )
            batch_size = len(image_names)
            for cam_id in camera_keys:
                cam_name = "cam" + cam_id
                if not os.path.exists(os.path.join(train_dir, seq, gesture, cam_name)):
                    continue
                if not os.path.exists(os.path.join(mask_dir, seq, gesture, cam_name)):
                    os.makedirs(os.path.join(mask_dir, seq, gesture, cam_name))
                R_cam = R[cam_id].to(device)
                t_cam = t[cam_id].to(device)
                t_cam = (-R_cam @ t_cam).view(1,3)
                K_cam = K[cam_id].to(device)
                output = mano_layer[hand_type](
                    global_orient=split_mano['mano_rot_pose'].reshape(-1, 3),
                    hand_pose=split_mano['mano_hand_pose'].reshape(-1, 45),
                    betas=split_mano['mano_shape'].reshape(-1, 10),
                    transl=split_mano['mano_trans'].reshape(-1, 3)
                )
                batch_size = len(image_names)
                # now_vertices = output.vertices
                now_vertices = torch.bmm(R_cam.expand(batch_size, 3, 3).to(device), output.vertices.transpose(2, 1)).transpose(2, 1) + t_cam.expand(batch_size, 1, 3).to(device)
                faces = torch.tensor(mano_layer[hand_type].faces.astype(np.int32)).repeat(batch_size, 1, 1).type_as(split_mano['mano_trans'])
                verts_rgb = torch.ones_like(mano_layer[hand_type].v_template).type_as(split_mano['mano_trans'])
                verts_rgb = verts_rgb.expand(batch_size, verts_rgb.shape[0], verts_rgb.shape[1])
                textures = TexturesVertex(verts_rgb)
                mesh = Meshes(
                    verts=now_vertices,
                    faces=faces,
                    textures=textures
                )
                cameras = cameras_from_opencv_projection(
                    R=torch.eye(3).repeat(batch_size, 1, 1).type_as(split_mano['mano_shape']),
                    tvec=torch.zeros(batch_size, 3).type_as(split_mano['mano_shape']),
                    # R = R,
                    # tvec= t,
                    camera_matrix=K_cam.expand(batch_size, 3, 3).type_as(split_mano['mano_shape']),
                    image_size=torch.tensor([height, width]).expand(batch_size, 2).type_as(split_mano['mano_shape'])
                ).to(split_mano['mano_trans'].device)
                raster_settings = RasterizationSettings(
                    image_size=(height, width),
                    faces_per_pixel=2,
                    perspective_correct=True,
                    blur_radius=0,
                )
                lights = PointLights(
                    location=[[0, 2, 0]],
                    diffuse_color=((0.5, 0.5, 0.5),),
                    specular_color=((0.5, 0.5, 0.5),)
                )
                render = MeshRenderer(
                    rasterizer=MeshRasterizer(raster_settings=raster_settings),
                    shader=(SoftPhongShader(lights=lights).to(split_mano['mano_trans'].device))
                )
                res = render(
                    mesh,
                    cameras=cameras,
                    # lights=lights
                )
                mask = res[..., 3:4]
                mask = torch.where(mask > 0.1, torch.ones_like(mask), torch.zeros_like(mask))
                mask = torch.cat([mask, mask, mask], dim=-1)
                # save meshs and masks
                for i in range(batch_size):
                    # mesh_path = os.path.join(output_dir,split,"mesh",str(i).zfill(8)+".ply")
                    mask_path =os.path.join(mask_dir, seq, gesture, cam_name,image_names[i])
                    # save_ply(mesh_path, now_vertices[i-start_idx], faces[i-start_idx])
                    cv2.imwrite(mask_path, mask[i].cpu().numpy()*255)

            print("finish seq: ", seq, "gesture: ", gesture)

    #     seq_dir = os.path.join(data_dir, seq)
    #     annot_path = os.path.join(seq_dir, annot_name)
    #     annot = json.load(open(annot_path, 'r'))
    #     K = [annot["camera_info"][camera_key]["K"]]
    #     K = torch.tensor(K, device=device)
    #     R = [annot["camera_info"][camera_key]["R"]]
    #     R = torch.tensor(R, device=device)
    #     t = [annot["camera_info"][camera_key]["T"]]
    #     t = torch.tensor(t, device=device)/1000.
        
    #     batch_size = 48
    #     frame_list = list(annot["manos"].keys())
    #     for i in range(528):
    #         if str(i) not in frame_list:
    #             print(f"frame {i} not in annot")
    #     print(f"frame_list length: {len(frame_list)}")
    #     if not os.path.exists(os.path.join(seq_dir, "mask",camera_key)):
    #         os.makedirs(os.path.join(seq_dir,"mask", camera_key))
    #     for frame_key in frame_list:
    #         manos["mano_trans"].append(annot["manos"][frame_key]["trans"])
    #         manos["mano_rot_pose"].append(annot["manos"][frame_key]["rot"])
    #         manos["mano_shape"].append(annot["manos"][frame_key]["shape"])
    #         manos["mano_hand_pose"].append(annot["manos"][frame_key]["hand_pose"])
    #     start_idx = 0
    #     end_idx = len(frame_list)
    #     while(True):
    #         if start_idx >= end_idx:
    #             break
    #         if start_idx + batch_size > end_idx:
    #             batch_size = end_idx - start_idx
    #         # print(f"start_idx:{start_idx}, end_idx:{end_idx}, batch_size:{batch_size}")
    #         current_idx = start_idx+batch_size
    #         split_mano = {'mano_trans':torch.tensor(manos["mano_trans"][start_idx:current_idx], device=device),
    #                     'mano_rot_pose':torch.tensor(manos["mano_rot_pose"][start_idx:current_idx], device=device),
    #                     'mano_shape':torch.tensor(manos["mano_shape"][start_idx:current_idx], device=device),
    #                     'mano_hand_pose':torch.tensor(manos["mano_hand_pose"][start_idx:current_idx], device=device),
    #             }
    #         # device = split_mano['mano_trans'].device
    #         # mano_layer[hand_type].to(device)
    #         output = mano_layer[hand_type](
    #             global_orient=split_mano['mano_rot_pose'].reshape(-1, 3),
    #             hand_pose=split_mano['mano_hand_pose'].reshape(-1, 45),
    #             betas=split_mano['mano_shape'].reshape(-1, 10),
    #             transl=split_mano['mano_trans'].reshape(-1, 3)
    #         )
    #         # now_vertices = output.vertices
    #         now_vertices = torch.bmm(R.expand(batch_size, 3, 3).to(device), output.vertices.transpose(2, 1)).transpose(2, 1) + t.expand(batch_size, 1, 3).to(device)
    #         faces = torch.tensor(mano_layer[hand_type].faces.astype(np.int32)).repeat(batch_size, 1, 1).type_as(split_mano['mano_trans'])
    #         verts_rgb = torch.ones_like(mano_layer[hand_type].v_template).type_as(split_mano['mano_trans'])
    #         verts_rgb = verts_rgb.expand(batch_size, verts_rgb.shape[0], verts_rgb.shape[1])
    #         textures = TexturesVertex(verts_rgb)
    #         mesh = Meshes(
    #             verts=now_vertices,
    #             faces=faces,
    #             textures=textures
    #         )
    #         cameras = cameras_from_opencv_projection(
    #             R=torch.eye(3).repeat(batch_size, 1, 1).type_as(split_mano['mano_shape']),
    #             tvec=torch.zeros(batch_size, 3).type_as(split_mano['mano_shape']),
    #             # R = R,
    #             # tvec= t,
    #             camera_matrix=K.expand(batch_size, 3, 3).type_as(split_mano['mano_shape']),
    #             image_size=torch.tensor([height, width]).expand(batch_size, 2).type_as(split_mano['mano_shape'])
    #         ).to(split_mano['mano_trans'].device)
    #         raster_settings = RasterizationSettings(
    #             image_size=(height, width),
    #             faces_per_pixel=2,
    #             perspective_correct=True,
    #             blur_radius=0,
    #         )
    #         lights = PointLights(
    #             location=[[0, 2, 0]],
    #             diffuse_color=((0.5, 0.5, 0.5),),
    #             specular_color=((0.5, 0.5, 0.5),)
    #         )
    #         render = MeshRenderer(
    #             rasterizer=MeshRasterizer(raster_settings=raster_settings),
    #             shader=(SoftPhongShader(lights=lights).to(split_mano['mano_trans'].device))
    #         )
    #         res = render(
    #             mesh,
    #             cameras=cameras,
    #             # lights=lights
    #         )
    #         mask = res[..., 3:4]
    #         mask = torch.where(mask > 0.1, torch.ones_like(mask), torch.zeros_like(mask))
    #         mask = torch.cat([mask, mask, mask], dim=-1)
    #         # save meshs and masks
    #         for i in range(start_idx, current_idx):
    #             # mesh_path = os.path.join(output_dir,split,"mesh",str(i).zfill(8)+".ply")
    #             mask_path = os.path.join(seq_dir,  "mask",camera_key, "image"+frame_list[i].zfill(4)+".jpg")
    #             # save_ply(mesh_path, now_vertices[i-start_idx], faces[i-start_idx])
    #             cv2.imwrite(mask_path, mask[i-start_idx].cpu().numpy()*255)
    #         start_idx = current_idx
        print(f"finish seq {seq}")
