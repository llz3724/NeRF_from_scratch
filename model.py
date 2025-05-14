import torch
import torch.nn as nn
class PositionEncoder(nn.Module):
    def __init__(self,
                 L,
                 input_dims=3):
        super().__init__()
        self.L=L
        self.input_dims=input_dims
        self.output_dims=self.input_dims
        if self.L>0:
            self.output_dims+=self.input_dims*2*self.L
        
        if self.L>0:
            # 不重复计算幂次
            bands=2.0**torch.repeat_interleave(torch.arange(float(self.L)),repeats=2)
            self.register_buffer('bands', bands, persistent=True)
        else:
            self.bands=None

    def forward(self, x):
        """
        x: (..., input_dims)
        """
        outputs = [x]

        if self.L>0:
            x_expand=x.unsqueeze(-1) # (...,input_dims,1)
            x_scaled=x_expand*self.bands*torch.pi
            # 建立索引，方便向量化操作
            indice_sin=torch.arange(0,2*self.L,step=2,device=x.device)
            indice_cos=torch.arange(1,2*self.L,step=2,device=x.device)
            encoded_values=torch.empty_like(x_scaled)
            encoded_values[...,indice_sin]=torch.sin(x_scaled[..., indice_sin])
            encoded_values[...,indice_cos]=torch.cos(x_scaled[..., indice_cos])
            batch_dims_shape=x.shape[:-1]
            encoded_features=encoded_values.reshape(*batch_dims_shape, self.input_dims * self.L * 2)
            outputs.append(encoded_features)

        return torch.cat(outputs, dim=-1)
    
class NeRF(nn.Module):
    def __init__(self,
                pos_xyz_dims,
                pos_dir_dims,
                hidden_dims=256,
                out_rgb_dims=3,
                out_sig_dims=1
                ):
        super().__init__()
        self.xyz_mlp1=nn.Sequential(
            nn.Linear(pos_xyz_dims,hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims),
            nn.ReLU()
        )
        # concencate
        self.xyz_mlp2=nn.Sequential(
            nn.Linear(pos_xyz_dims+hidden_dims,hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims,hidden_dims)
        )

        self.out_sig=nn.Sequential(
            nn.Linear(hidden_dims,out_sig_dims),
            nn.ReLU(),
        )

        self.feature_encode=nn.Linear(hidden_dims,hidden_dims)

        self.out_rgb=nn.Sequential(
            nn.Linear(hidden_dims+pos_dir_dims,hidden_dims//2),
            nn.ReLU(),
            nn.Linear(hidden_dims//2,3),
            nn.Sigmoid()
        )

        # fine sample 和 coarse sample 网络结构相同但权重不一样，学习目标不一样

    def forward(self,encode_xyz,encode_dir):
        """
        input:
            encode_syz:
            encode_dir:
        output:
            rgb: 
            sigma: 
        """
        xyz_1=self.xyz_mlp1(encode_xyz)
        xyz_feature=self.xyz_mlp2(torch.cat([xyz_1,encode_xyz],dim=-1))
        xyz_encode_final=self.feature_encode(xyz_feature)
        rgb=self.out_rgb(torch.cat([xyz_encode_final,encode_dir],dim=-1))
        sigma=self.out_sig(xyz_feature)
        
        return rgb,sigma