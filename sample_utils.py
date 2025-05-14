import torch
def dep_to_pos(start,direction,dep):
    # (N_rays,1,3)+(N_rays,N_coarse_sample,1)*(N_rays,1,3)
    # 自动先广播大小为1的维度
    return start.unsqueeze(1)+dep.unsqueeze(2)*direction.unsqueeze(1)

def coarse_sample(start,direction,near,far,N_coarse_sample,perturb=True,device='cuda'):
    """
    input:
        start: (N_rays,3)
        direction: (N_rays,3), unit vectors
        near: scalar
        far: scalar
    output:
        sample_dep: (N_rays,N_coarse_sample)
    """
    N_rays=start.shape[0]
    
    dep_arr_bin_starts=torch.linspace(near,far,N_coarse_sample+1,device=device)[:-1] # (N_coarse_sample)
    
    dep_arr_expanded = dep_arr_bin_starts.unsqueeze(0).expand((N_rays,N_coarse_sample))
    dep_arr = dep_arr_expanded.clone() 
    
    gap=(far-near)/N_coarse_sample
    
    if perturb:
        offset=torch.rand((N_rays,N_coarse_sample),device=device)*gap
        dep_arr+=offset
    else:
        dep_arr += (gap*0.5)
    return dep_arr

def invert_trans(coarse_dep,weights,N_sample,perturb=True,device='cuda'):
    """
    input:
        coarse_dep: (N_rays,N_pos)
        weights: (N_rays, N_pos)

    output:
        sample_dep: (N_rays,N_sample)
    """
    N_rays,N_pos=coarse_dep.shape
    eps=1e-5
    weights=weights+eps
    pdf=weights/torch.sum(weights,dim=-1,keepdim=True)
    cdf=torch.cumsum(pdf,dim=-1)
    zero_prefix=torch.zeros_like(cdf[...,:1],device=device)
    full_cdf=torch.cat([zero_prefix,cdf],dim=-1)
    
    if perturb:
        u_base=torch.linspace(0.0,1.0,steps=N_sample+1,device=device)[:-1]
        offset=torch.rand((N_rays,N_sample),device=device)/N_sample
        u=u_base.unsqueeze(0)+offset
    else:
        u=torch.linspace(0.0,1.0,steps=N_sample+1,device=device)[:-1]+0.5/N_sample
        if N_sample>0:
            u=u.unsqueeze(0).expand((N_rays,N_sample))
        else:
            u=torch.empty((N_rays,0),device=device)
    u=u.contiguous()
    u=torch.clamp(u,0.0,1.0-eps)

    bin_k_idx=torch.searchsorted(full_cdf,u,right=True)-1
    bin_k_idx=torch.clamp(bin_k_idx,0,N_pos-1)

    cdf_low=torch.gather(full_cdf,-1,bin_k_idx)
    cdf_high=torch.gather(full_cdf,-1,bin_k_idx+1)

    clp_idx=torch.clamp(bin_k_idx+1,0,N_pos-1)
    dep_low=torch.gather(coarse_dep,-1,bin_k_idx)
    dep_high=torch.gather(coarse_dep,-1,clp_idx)
    denom=cdf_high-cdf_low
    denom=torch.max(denom,torch.full_like(denom,eps))

    t=(u-cdf_low)/denom
    t=torch.clamp(t,0.0,1.0)
    sample_dep=dep_low+t*(dep_high-dep_low)

    return sample_dep

def fine_sample(coarse_dep,coarse_position,weights,N_fine_sample,perturb=True,device='cuda'):
    """
    input:
        coarse_dep: 
        coarse_position: 
        weights: 

    output:
        fine_dep:
    """
    eps=1e-5
    N_rays=coarse_position.shape[0]
    N_coarse_sample=coarse_position.shape[1]

    new_dep=invert_trans(coarse_dep,weights,N_fine_sample,perturb,device)

    comb_dep=torch.cat([new_dep,coarse_dep],dim=-1)
    fine_dep,_=torch.sort(comb_dep,dim=-1)

    return fine_dep