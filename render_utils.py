import torch

def cumprod_exclusive(input):
    ones=torch.ones_like(input[...,:1])
    padded=torch.cat([ones,input[...,:-1]],dim=-1)
    return torch.cumprod(padded,dim=-1)

def volume_render(nerf_rgb,nerf_sigma,z_vals,white_bkgd=True):
    """
    input:
        nerf_rgb: (N_rays,N_sample,3)
        nerf_sigma: (N_rays,N_sample)
        z_vals: (N_rays,N_sample)
    output:
        final_rgb: (N_rays,3)
        final_dep: (N_rays,)
        final_acc: (N_rays,)
        weights: (N_rays,N_sample)
    """
    N_rays=nerf_rgb.shape[0]
    N_sample=nerf_rgb.shape[1]
    deltas=torch.zeros((N_rays,N_sample),device=z_vals.device,dtype=z_vals.dtype)
    deltas[...,:-1]=z_vals[...,1:]-z_vals[...,:-1]
    deltas[...,-1]=torch.finfo(deltas.dtype).max

    alpha=1-torch.exp(-nerf_sigma*deltas)

    eps=1e-10
    transmittance=cumprod_exclusive(1.0-alpha+eps)

    # 方便精细采样
    weights=transmittance*alpha

    final_rgb=torch.sum(weights.unsqueeze(-1)*nerf_rgb,dim=1)
    final_dep=torch.sum(weights*z_vals,dim=1)
    final_acc=torch.sum(weights,dim=-1)
    if white_bkgd:
        final_rgb=final_rgb+(1.0-final_acc.unsqueeze(-1))
    
    return final_rgb,final_dep,final_acc,weights

