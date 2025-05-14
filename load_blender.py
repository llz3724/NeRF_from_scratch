import os
import torch
import numpy as np
import imageio.v2 as imageio
import json
import torchvision.transforms.functional as TF

# 需要人为构造新视角才用到
# 平移
# def _trans_t(t):
#     return torch.tensor([
#         [1,0,0,0],
#         [0,1,0,0],
#         [0,0,1,t],
#         [0,0,0,1]],dtype=torch.float32)

# def _rot_phi(phi):
#     phi_rad=phi*np.pi/180.0
#     return torch.tensor([
#         [1,0,0,0],
#         [0,np.cos(phi_rad),-np.sin(phi_rad),0],
#         [0,np.sin(phi_rad),np.cos(phi_rad),0],
#         [0,0,0,1]], dtype=torch.float32)

# def _rot_theta(th):
#     th_rad=th*np.pi/180.
#     return torch.tensor([
#         [np.cos(th_rad),0,-np.sin(th_rad),0],
#         [0,1,0,0],
#         [np.sin(th_rad),0,np.cos(th_rad),0],
#         [0,0,0,1]],dtype=torch.float32)

# def pose_spherical(theta,phi,radius):
#     c2w=_trans_t(radius)
#     c2w=_rot_phi(phi)@c2w
#     c2w=_rot_theta(theta)@c2w
#     c2w=torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=torch.float32)@c2w
#     return c2w

class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,basedir,split='train',half_res=False,white_bkgd=True,near=2.0,far=6.0,device='cuda'):
        super().__init__()
        self.basedir=basedir
        self.split=split
        self.half_res=half_res
        self.white_bkgd=white_bkgd
        self.near=near
        self.far=far
        self.device=device 

        #首先加载所有元数据，因为计算焦距等可能需要训练集的相机参数
        metas={}
        for s_meta in ['train','val','test']:
            meta_path=os.path.join(basedir,f'transforms_{s_meta}.json')
            if os.path.exists(meta_path):
                with open(meta_path,'r') as fp:
                    metas[s_meta]=json.load(fp)
            elif s_meta=='train':
                raise FileNotFoundError(f"Train metadata transforms_train.json not found in {basedir}")

        current_meta=metas[self.split]
        
        # 从train的元数据中获取相机内参相关的全局信息
        # 即使当前split不是train，通常也用train的相机参数作为参考
        train_meta_for_focal=metas.get('train',current_meta)
        
        camera_angle_x=float(train_meta_for_focal['camera_angle_x'])
        # 原始图像的 H, W 在元数据中没有直接给出，需要从第一张图像读取
        temp_fname=os.path.join(self.basedir,train_meta_for_focal['frames'][0]['file_path']+'.png')
        temp_img_shape=imageio.imread(temp_fname).shape
        H_orig,W_orig=temp_img_shape[0],temp_img_shape[1]

        self.focal=0.5*W_orig/np.tan(0.5*camera_angle_x)
        self.H,self.W=H_orig,W_orig

        if self.half_res:
            self.H=H_orig//2
            self.W=W_orig//2
            self.focal=self.focal/2.0

        current_imgs_list=[]
        current_poses_list=[]

        print(f"Loading data for split: {self.split}")
        for frame in current_meta['frames']:
            fname=os.path.join(self.basedir,frame['file_path']+'.png')
            img_np=imageio.imread(fname) # (H_orig, W_orig, 4)
            
            # 转换为Tensor并移到设备
            img_tensor_rgba=torch.from_numpy(img_np).to(self.device).float()/255.0

            if self.half_res:
                # (H, W, C) -> (C, H, W) for TF.resize
                img_tensor_chw=img_tensor_rgba.permute(2,0,1).unsqueeze(0)
                img_resized_chw=TF.resize(img_tensor_chw,size=[self.H,self.W],antialias=True)
                # antialias=True 是抗锯齿
                img_tensor_rgba=img_resized_chw.squeeze(0).permute(1,2,0) # Back to (H, W, C)
            
            current_imgs_list.append(img_tensor_rgba)
            current_poses_list.append(torch.tensor(frame['transform_matrix'],dtype=torch.float32, device=self.device))
        
        imgs_rgba_stacked=torch.stack(current_imgs_list,0) # (N_split, H, W, 4)
        self.poses=torch.stack(current_poses_list,0)    # (N_split, 4, 4)

        # RGBA to RGB
        rgb=imgs_rgba_stacked[..., :3]
        alpha=imgs_rgba_stacked[..., 3:4] 
        if self.white_bkgd:
            self.images=rgb*alpha+(1.0-alpha) # alpha合成到rgb的方式，全0是黑色，全1是白色
        else:
            self.images=rgb # (N_split, H, W, 3)

        self.K=torch.tensor([
            [self.focal,0,0.5*self.W],
            [0,self.focal,0.5*self.H],
            [0,0,1]
        ],dtype=torch.float32,device=self.device)

        # # 生成新视角，独立于train/val/test，如果有生成视频的需求还是需要的
        # self.render_poses = torch.stack(
        #     [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0
        # ).to(self.device)

        # 预处理出所有像素对应的光线
        print(f"Precomputing rays for '{self.split}' split on device '{self.device}'...")
        rays_o_list=[]
        rays_d_list=[]
        target_s_list=[]
        for i in range(self.images.shape[0]):
            rays_o,rays_d=self._get_rays(self.H,self.W,self.K,self.poses[i])
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
            target_s_list.append(self.images[i].reshape(-1, 3)) 
        
        self.all_rays_o=torch.cat(rays_o_list, 0)    # (N_imgs_split * H * W, 3)
        self.all_rays_d=torch.cat(rays_d_list, 0)    # (N_imgs_split * H * W, 3)
        self.all_target_s=torch.cat(target_s_list, 0) # (N_imgs_split * H * W, 3)
        # 将所有图片的光线混合，原论文做法
        print("Ray precomputation done.")

    @staticmethod
    def _get_rays(H,W,K,c2w,device=None):
        if device is None:
            device=K.device

        j,i=torch.meshgrid(torch.linspace(0,H-1,H,device=device),torch.linspace(0,W-1,W,device=device),indexing='ij')
        i=i.float()
        j=j.float()

        dirs=torch.stack([(i-K[0,2])/K[0,0],-(j-K[1,2])/K[1,1],-torch.ones_like(i)],dim=-1) 
        
        rays_d=torch.matmul(dirs,c2w[:3,:3].T) 
        rays_o=c2w[:3,-1].expand_as(rays_d)
        
        return rays_o.reshape(-1,3),rays_d.reshape(-1,3)

    def __len__(self):
        if self.split == 'train':
            return self.all_target_s.shape[0] if hasattr(self, 'all_target_s') else 0
        else: # 'val' or 'test'
            return self.images.shape[0] if hasattr(self, 'images') else 0

    def __getitem__(self, idx):
        if self.split=='train':
            if not hasattr(self, 'all_target_s') or self.all_target_s.shape[0] == 0:
                raise IndexError(f"Train dataset is empty or rays not precomputed for split '{self.split}'.")
            return {
                'rays_o': self.all_rays_o[idx],
                'rays_d': self.all_rays_d[idx],
                'target_s': self.all_target_s[idx]
            }
        else:
            if not hasattr(self,'images') or self.images.shape[0]==0:
                raise IndexError(f"Dataset is empty for split '{self.split}'.")
            
            pose=self.poses[idx]
            target_img_rgb=self.images[idx]

            # _get_rays 会使用 K 和 pose 所在的设备
            rays_o, rays_d=self._get_rays(self.H,self.W,self.K,pose)
            
            return {
                'rays_o': rays_o,                # (H*W, 3)
                'rays_d': rays_d,                # (H*W, 3)
                'target_rgbs': target_img_rgb.reshape(-1, 3), # (H*W, 3)
                'H': self.H,
                'W': self.W,
                'c2w': pose                      # (4,4) for reference
            }
