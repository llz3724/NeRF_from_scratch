import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio
from tqdm import tqdm,trange

from load_blender import BlenderDataset
from model import PositionEncoder, NeRF
import sample_utils
import render_utils

def load_config(args):
    with open(args.config,'r') as f:
        config=json.load(f)
    return config

def psnr_metric(img_true,img_pred):
    mse=torch.mean((img_true-img_pred)**2)
    if mse<1e-10:
        return float('inf')
    return -10.0*torch.log10(mse)

def render_full_image(H,W,K,c2w_pose,near,far,N_coarse,N_fine,perturb_coarse,perturb_fine,
                      pos_enc_xyz,pos_enc_dir,model_coarse,model_fine,
                      white_bkgd,device,chunk_size):
    """
    渲染完整图片
    """
    
    #预生成所有光线
    rays_o_img,rays_d_img=BlenderDataset._get_rays(H,W,K,c2w_pose,device=device) #(H*W, 3)

    rendered_rgb_parts=[]
    for i in range(0,rays_o_img.shape[0],chunk_size):
        rays_o_batch=rays_o_img[i:i+chunk_size]
        rays_d_batch=rays_d_img[i:i+chunk_size]

        if not rays_o_batch.numel(): continue

        _,rgb_fine,_,_=render_rays_batched(
            rays_o_batch,rays_d_batch,near,far,N_coarse,N_fine,
            perturb_coarse,perturb_fine,
            pos_enc_xyz,pos_enc_dir,model_coarse,model_fine,
            white_bkgd,device,chunk_size # Pass chunk_size for MLP query if needed
        )
        rendered_rgb_parts.append(rgb_fine)
    
    img_rendered=torch.cat(rendered_rgb_parts,0).reshape(H,W,3)
    return img_rendered

def render_rays_batched(
    rays_o,rays_d,near,far,N_coarse,N_fine,perturb_coarse,perturb_fine,
    position_encoder_xyz,position_encoder_dir,model_coarse,model_fine,
    white_bkgd,device,mlp_chunk_size):
    """
    渲染一批光线
    rays_o, rays_d: (N_batch_rays, 3)
    """
    N_batch_rays = rays_o.shape[0]

    #Coarse Sampling
    coarse_dep=sample_utils.coarse_sample(
        rays_o,rays_d,near,far,N_coarse,perturb=perturb_coarse,device=device
    ) 
    coarse_position=sample_utils.dep_to_pos(rays_o, rays_d, coarse_dep)

    coarse_position_flat=coarse_position.reshape(-1, 3) # (N_batch_rays * N_coarse, 3)
    dirs_expanded_coarse=rays_d.unsqueeze(1).expand_as(coarse_position)
    dirs_flat_coarse=dirs_expanded_coarse.reshape(-1, 3)

    encoded_xyz_coarse=position_encoder_xyz(coarse_position_flat)
    encoded_dir_coarse=position_encoder_dir(dirs_flat_coarse)

    # Query coarse model (chunked for MLP)
    rgb_raw_coarse_list=[]
    sigma_raw_coarse_list=[]
    for i in range(0,encoded_xyz_coarse.shape[0],mlp_chunk_size):
        rgb_c,sigma_c=model_coarse(
            encoded_xyz_coarse[i:i+mlp_chunk_size],
            encoded_dir_coarse[i:i+mlp_chunk_size]
        )
        rgb_raw_coarse_list.append(rgb_c)
        sigma_raw_coarse_list.append(sigma_c)
    
    rgb_raw_coarse_flat=torch.cat(rgb_raw_coarse_list,0)
    sigma_raw_coarse_flat=torch.cat(sigma_raw_coarse_list,0)

    # rgb_sigma_coarse (N_batch_rays, N_coarse, 4)
    # 最后是 sigma
    rgb_sigma_coarse=torch.cat([rgb_raw_coarse_flat,sigma_raw_coarse_flat],dim=-1)
    rgb_sigma_coarse=rgb_sigma_coarse.reshape(N_batch_rays,N_coarse,4)

    rgb_map_coarse,dep_map_coarse,acc_map_coarse,weights_coarse=render_utils.volume_render(
        rgb_sigma_coarse[...,:3],    # (N_rays, N_sample, 3)
        rgb_sigma_coarse[...,3],     # (N_rays, N_sample)
        coarse_dep,                # (N_rays, N_sample)
        white_bkgd=white_bkgd
    ) 
    # weights_coarse: (N_batch_rays, N_coarse)

    #Fine Sampling
    fine_dep_sorted=sample_utils.fine_sample(
        coarse_dep=coarse_dep,
        coarse_position=coarse_position,
        weights=weights_coarse.detach(), #detach是复制一个没有梯度信息的副本，如果没有会试图把粗糙网络作为精细网络的上游导致反向传播不合预期
        N_fine_sample=N_fine,
        perturb=perturb_fine, 
        device=device
    )
    
    z_vals_combined_sorted=fine_dep_sorted
    pts_fine_combined=sample_utils.dep_to_pos(rays_o, rays_d, z_vals_combined_sorted)

    pts_fine_flat=pts_fine_combined.reshape(-1,3)
    dirs_expanded_fine=rays_d.unsqueeze(1).expand_as(pts_fine_combined)
    dirs_flat_fine=dirs_expanded_fine.reshape(-1,3)

    encoded_xyz_fine=position_encoder_xyz(pts_fine_flat)
    encoded_dir_fine=position_encoder_dir(dirs_flat_fine)

    rgb_raw_fine_list=[]
    sigma_raw_fine_list=[]
    for i in range(0, encoded_xyz_fine.shape[0],mlp_chunk_size):
        rgb_f,sigma_f=model_fine(
            encoded_xyz_fine[i:i+mlp_chunk_size],
            encoded_dir_fine[i:i+mlp_chunk_size]
        )
        rgb_raw_fine_list.append(rgb_f)
        sigma_raw_fine_list.append(sigma_f)

    rgb_raw_fine_flat=torch.cat(rgb_raw_fine_list,0)
    sigma_raw_fine_flat=torch.cat(sigma_raw_fine_list,0)
    
    rgb_sigma_fine=torch.cat([rgb_raw_fine_flat,sigma_raw_fine_flat],dim=-1)
    rgb_sigma_fine=rgb_sigma_fine.reshape(N_batch_rays,N_coarse+N_fine,4)
    
    rgb_map_fine,disp_map_fine,acc_map_fine,weights_fine=render_utils.volume_render(
        rgb_sigma_fine[...,:3], 
        rgb_sigma_fine[...,3], 
        z_vals_combined_sorted, 
        white_bkgd=white_bkgd
    )

    return rgb_map_coarse,rgb_map_fine,acc_map_coarse,acc_map_fine

def train():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=str,required=True,help='Path to hparams.json config file')

    args=parser.parse_args()

    config=load_config(args)
    
    if config['device']=='cuda' and torch.cuda.is_available():
        device=torch.device(f"cuda:{config['gpu_id']}")
        torch.cuda.set_device(device)
    else:
        device=torch.device('cpu')
    print(f"Using device: {device}")

    logdir_exp=os.path.join(config['logdir'],config['experiment_name'])
    os.makedirs(logdir_exp,exist_ok=True)
    os.makedirs(os.path.join(logdir_exp,'imgs_test'),exist_ok=True)
    os.makedirs(os.path.join(logdir_exp,'imgs_val'),exist_ok=True)

    #保存当前用的超参，因为每次实验用的超参可能不同，总目录超参用于当前实验
    with open(os.path.join(logdir_exp, 'config.json'), 'w') as f:
        json.dump(config,f,indent=4)

    print(f"Loading dataset from: {config['basedir']}")
    train_dataset=BlenderDataset(
        basedir=config['basedir'],
        split='train',
        half_res=config['half_res_dataset'],
        white_bkgd=config['white_bkgd_dataset'],
        device=device
    )
    
    val_dataset=BlenderDataset(
        basedir=config['basedir'],
        split='val',
        half_res=config['half_res_dataset'],
        white_bkgd=config['white_bkgd_dataset'],
        device=device
    )
    #找一张val快速看效果
    val_render_idx=0
    val_render_data=val_dataset[val_render_idx] if len(val_dataset) > 0 else None

    pos_encoder_xyz=PositionEncoder(L=config['L_xyz'],input_dims=3).to(device)
    pos_encoder_dir=PositionEncoder(L=config['L_dir'],input_dims=3).to(device)
    
    model_coarse=NeRF(
        pos_xyz_dims=pos_encoder_xyz.output_dims,
        pos_dir_dims=pos_encoder_dir.output_dims,
        hidden_dims=config['mlp_hidden_dims']
    ).to(device)
    model_fine=NeRF(
        pos_xyz_dims=pos_encoder_xyz.output_dims,
        pos_dir_dims=pos_encoder_dir.output_dims,
        hidden_dims=config['mlp_hidden_dims']
    ).to(device)

    params=list(model_coarse.parameters())+list(model_fine.parameters())
    optimizer=optim.Adam(params,lr=config['lr_init'])
    
    scheduler=optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(config['lr_decay_rate']**(1/config['lr_decay_steps']))
    )

    start_iter=0
    if config['load_checkpoint_path'] and os.path.exists(config['load_checkpoint_path']):
        print(f"Loading checkpoint from {config['load_checkpoint_path']}")
        checkpoint=torch.load(config['load_checkpoint_path'],map_location=device)
        model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
        model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter=checkpoint['iteration'] + 1
        print(f"Resuming from iteration {start_iter}")

    if not hasattr(train_dataset,'all_rays_o') or train_dataset.all_rays_o.shape[0]==0:
        print("ERROR: Training rays not precomputed or dataset is empty. Exiting.")
        return

    all_train_rays_o=train_dataset.all_rays_o
    all_train_rays_d=train_dataset.all_rays_d
    all_train_targets=train_dataset.all_target_s

    print(f"Starting training from iteration {start_iter} to {config['num_iters']-1}")
    pbar = trange(start_iter, config['num_iters'], leave=True)
    for i in pbar:
        #采样
        ray_indices=torch.randint(0,all_train_rays_o.shape[0],(config['batch_size'],),device=device)
        batch_rays_o=all_train_rays_o[ray_indices]
        batch_rays_d=all_train_rays_d[ray_indices]
        batch_targets=all_train_targets[ray_indices]

        #渲染
        model_coarse.train()
        model_fine.train()
        rgb_coarse,rgb_fine,_,_=render_rays_batched(
            batch_rays_o, batch_rays_d,
            config['near_plane'],config['far_plane'],
            config['N_coarse'],config['N_fine'],
            config['perturb_coarse'],config['perturb_fine'],
            pos_encoder_xyz,pos_encoder_dir,
            model_coarse, model_fine,
            config['white_bkgd_dataset'],device,
            config['chunk_size']
        )

        # Loss
        loss_coarse=torch.mean((rgb_coarse-batch_targets)**2)
        loss_fine=torch.mean((rgb_fine-batch_targets)**2)
        total_loss=loss_coarse+loss_fine

        # Optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        i_weights=int(config['num_iters']*config['i_weights_per'])

        if i%config['i_print']==0 or i==config['num_iters']-1:
            psnr_fine=psnr_metric(batch_targets, rgb_fine)
            pbar.set_description(
                f"[TRAIN] Iter: {i}/{config['num_iters']-1} "
                f"Loss: {total_loss.item():.4f} PSNR: {psnr_fine.item():.2f} "
                f"LR: {scheduler.get_last_lr()[0]:.1e}"
            )

        if i%i_weights==0 and i>0:
            path=os.path.join(logdir_exp,f'iter_{i:06d}.tar')
            torch.save({
                'iteration':i,
                'model_coarse_state_dict':model_coarse.state_dict(),
                'model_fine_state_dict':model_fine.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'config':config
            },path)
            print(f"Saved checkpoint at {path}")

        if i%config['i_img']==0 and i>0:
            model_coarse.eval()
            model_fine.eval()
            with torch.no_grad():
                K_val=val_dataset.K
                c2w_val=val_render_data['c2w']
                H_val,W_val=val_dataset.H,val_dataset.W
                
                img_rendered=render_full_image(
                    H_val,W_val,K_val,c2w_val,
                    config['near_plane'],config['far_plane'],
                    config['N_coarse'],128,
                    False,False,
                    pos_encoder_xyz,pos_encoder_dir,model_coarse,model_fine,
                    config['white_bkgd_dataset'],device,config['chunk_size']
                )
            img_rendered_np=(img_rendered.cpu().numpy()*255).astype(np.uint8)
            imageio.imwrite(os.path.join(logdir_exp,'imgs_val',f'{i:06d}.png'),img_rendered_np)
            print(f"Rendered validation image at iteration {i}")
            
            target_img_val=val_render_data['target_rgbs'].reshape(H_val,W_val,3)
            psnr_val_img=psnr_metric(target_img_val.to(device),img_rendered)
            print(f"Validation image PSNR: {psnr_val_img.item():.2f}")

    print("Training finished.")

    path=os.path.join(logdir_exp,'final.tar')
    torch.save({
        'iteration':config['num_iters']-1,
        'model_coarse_state_dict':model_coarse.state_dict(),
        'model_fine_state_dict':model_fine.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
        'config':config
    },path)
    print(f"Saved final model at {path}")


if __name__=='main':
    train()

    # 
