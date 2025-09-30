from .FluidGPT_B import *

def gen_t_embedding(t, emb_dim, max_positions=10000):
    t = t * max_positions
    half_dim = emb_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    return emb

class FluidGPT_FM(nn.Module):
    def __init__(self, emb_dim=96, data_dim=[64,3,2,128,128], patch_size=(8,8), hiddenout_dim=128, flowmatching_emb_dim=256,
                 depth=2, stage_depths=[6,6,10,6,6], num_heads=[6,6,12,6,6], window_size=4, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., use_flex_attn=True, norm_layer=nn.LayerNorm,
                 act=nn.GELU, skip_connect=ConvNeXtBlock, gradient_flowthrough=[True, False, False]):
        super().__init__()

        # assert that every element in stage_depths is divisible by 3 except for the middle element
        assert all(stage_depths[i] % 3 == 0 for i in range(len(stage_depths)) if i != depth), "stage depth must be divisible by 3 at non-middle elements"
        assert stage_depths[depth] % 2 == 0, "stage depth must be divisible by 2 at middle element"
        self.embedding = LinearEmbedding(emb_dim, data_dim, patch_size, hiddenout_dim, act)
        self.pos_encoding = SpatiotemporalPositionalEncoding(emb_dim, data_dim[3] // patch_size[0], data_dim[4] // patch_size[1], data_dim[1])
        #print(data_dim[2] // patch_size[0], data_dim[2] // patch_size[0], data_dim[1])
        self.blockDown = nn.ModuleList(nn.ModuleList() for i in range(depth))
        self.blockMiddle = nn.ModuleList()
        self.blockUp = nn.ModuleList(nn.ModuleList() for i in range(depth))
        self.patchMerges = nn.ModuleList()
        self.patchUnmerges = nn.ModuleList()
        self.skip_connects = nn.ModuleList()
        # TODO: implement act
         
        self.flowmatching_emb_dim = flowmatching_emb_dim
        self.flowt_proj = nn.Linear(flowmatching_emb_dim, flowmatching_emb_dim) 
        self.flowt_act = nn.SiLU()
        #self.flowt_proj_easy = nn.Linear(1, emb_dim) 
        self.flowt_proj2 = nn.Linear(flowmatching_emb_dim, flowmatching_emb_dim) #2**depth * emb_dim

        self.flow_d1 = nn.Linear(flowmatching_emb_dim, 32)
        self.flow_d2 = nn.Linear(flowmatching_emb_dim, 64)
        self.flow_d3 = nn.Linear(flowmatching_emb_dim, 128)
        self.flow_d4 = nn.Linear(flowmatching_emb_dim, 64)
        self.flow_d5 = nn.Linear(flowmatching_emb_dim, 32)

        self.depth = depth
        self.middleblocklen = stage_depths[depth]
        self.gradient_flowthrough = gradient_flowthrough
        self.skip_connect = skip_connect

        self.resnorms_down = nn.ModuleList(nn.ModuleList() for _ in range(depth))
        self.resnorms_middle = nn.ModuleList()
        self.resnorms_up = nn.ModuleList(nn.ModuleList() for _ in range(depth))

        for i in range(depth):
            patch_grid_res = (data_dim[3] // (patch_size[0] * 2**i), data_dim[4] // (patch_size[1] * 2**i))
            for j in range(stage_depths[i]):

                if j % 3 == 0:
                    self.resnorms_down[i].append(nn.LayerNorm((emb_dim + 32) * 2**i))
                    self.blockDown[i].append(
                        SwinStage(
                            (emb_dim + 32) * 2**i, 
                            patch_grid_res=patch_grid_res, 
                            num_heads=num_heads[i], 
                            window_size=window_size, 
                            mlp_ratio = mlp_ratio, 
                            qkv_bias = qkv_bias, 
                            drop=drop,
                            attn_drop = attn_drop,
                            use_flex_attn = use_flex_attn, 
                            act=act, 
                            norm_layer=norm_layer
                        )
                    )
                    #j += 1
                
                if j % 3 == 2:
                    self.blockDown[i].append(
                        TemporalBlock(
                            emb_dim=(emb_dim + 32) * 2**i,
                            num_heads=num_heads[i],
                            max_timesteps=data_dim[1],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop,
                            attn_drop=attn_drop,
                            use_flex_attn=use_flex_attn,
                            act_layer=act,
                            norm_layer=norm_layer
                        )
                    )
                
                
            self.patchMerges.append(PatchMerge(emb_dim * 2**i))


        patch_grid_res = (data_dim[3] // (patch_size[0] * 2**depth), data_dim[4] // (patch_size[1] * 2**depth))
        full_window_size = data_dim[3] // (patch_size[0] * 2**depth)
        #print('full_window_size', full_window_size)
        for i in range(stage_depths[depth]):
            
            if i % 2 == 0:
                self.resnorms_middle.append(nn.LayerNorm((emb_dim + 32) * 2**depth))
                self.blockMiddle.append(
                    SpatialSwinBlock(
                        (emb_dim + 32) * 2**depth,
                        patch_grid_res=patch_grid_res,
                        num_heads=num_heads[depth],
                        window_size = full_window_size,
                        shift_size=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=0.,
                        attn_drop=0.,
                        use_flex_attn=use_flex_attn,
                        act=act,
                        norm_layer=norm_layer
                    )
                )
            
            if i % 2 == 1:
                self.blockMiddle.append(
                    TemporalBlock(
                        emb_dim=(emb_dim + 32) * 2**depth,
                        num_heads=num_heads[depth],
                        max_timesteps=data_dim[1],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        use_flex_attn=use_flex_attn,
                        act_layer=act,
                        norm_layer=norm_layer
                    )
                )


        for i in reversed(range(depth)):
            #print(i)
            patch_grid_res = (data_dim[3] // (patch_size[0] * 2**i), data_dim[4] // (patch_size[1] * 2**i))
            for j in range(stage_depths[depth + i + 1]):
                #print(depth + i, j)
                #print(i, emb_dim * 2**i)
                
                if j % 3 == 0:
                    self.resnorms_up[depth - i - 1].append(nn.LayerNorm((emb_dim + 32) * 2**i))
                    self.blockUp[depth - i - 1].append(
                        SwinStage(
                            (emb_dim + 32) * 2**i,
                            patch_grid_res=patch_grid_res, 
                            num_heads=num_heads[2*depth - i], 
                            window_size=window_size, 
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop,
                            attn_drop=attn_drop,
                            use_flex_attn=use_flex_attn,
                            act=act,
                            norm_layer=norm_layer
                        )
                    )
                    #j += 1
                
                if j % 3 == 2:
                    self.blockUp[depth - i - 1].append(
                        TemporalBlock(
                            emb_dim=(emb_dim + 32) * 2**i,
                            num_heads=num_heads[2*depth - i],
                            max_timesteps=data_dim[1],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop=drop,
                            attn_drop=attn_drop,
                            use_flex_attn=use_flex_attn,
                            act_layer=act,
                            norm_layer=norm_layer
                        )
                    )
                
                    
            self.patchUnmerges.append(PatchUnMerge(emb_dim * 2**(i+1)))
            self.skip_connects.append(skip_connect(emb_dim * 2**i)) if skip_connect is not None else None

    def forward(self, x, t, extra={None}):
        # shape checks
        x = x.permute(0,2,1,3,4).contiguous()
        #print('\nstarting pred...')
        if x.ndim != 5:
            raise ValueError(f"Input tensor must be 5D, but got {x.ndim}D")
        if x.shape[1] != self.embedding.T or x.shape[2] != self.embedding.C or x.shape[3] != self.embedding.H or x.shape[4] != self.embedding.W:
            raise ValueError(f"Input tensor must be of shape (B, {self.embedding.T}, {self.embedding.C}, {self.embedding.H}, {self.embedding.W}), but got {x.shape}")
        
        
        skips = []
        #print('module_list', self.blockDown)
        #print('block up', self.blockUp)
        x = self.embedding.encode(x, proj=True)
        x = self.pos_encoding(x)

        # flowmatching stuff
        #t = gen_t_embedding(t, self.flowmatching_emb_dim)
        #t1 = self.flowt_proj(t)
        #t1 = t1.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2], 1)  
        #x = x + t1
        #t1 = self.flowt_proj_easy(t.unsqueeze(-1))
        #print(t.shape)
        t = gen_t_embedding(t, self.flowmatching_emb_dim)
        #print(t.shape)
        t0 = self.flowt_proj(t)
        t0 = self.flowt_act(t0)
        t0 = self.flowt_proj2(t0)

        t1 = self.flowt_act(t0)
        t1 = self.flow_d1(t1)
        t1 = t1.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2], 1)

        t2 = self.flowt_act(t0)
        t2 = self.flow_d2(t2)
        t2 = t2.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2] // 4, 1)

        t3 = self.flowt_act(t0)
        t3 = self.flow_d3(t3)
        t3 = t3.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2] // 16, 1)
        
        t4 = self.flowt_act(t0)
        t4 = self.flow_d4(t4)
        t4 = t4.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2] // 4, 1)

        t5 = self.flowt_act(t0)
        t5 = self.flow_d5(t5)
        t5 = t5.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2], 1)

        # ===== DOWN =====
        for i, module_list in enumerate(self.blockDown):
            for j, module in enumerate(module_list):
                #print(j)
                if j % 2 == 0:
                    residual = x if self.gradient_flowthrough[0] else None
                    #print(x.shape, t1.shape, t2.shape)
                    if i == 0:
                        x = torch.concat((x, t1), dim=3)
                    else:
                        x = torch.concat((x, t2), dim=3)
                    #print('after concat', x.shape)
                    x = self.resnorms_down[i][j // 2](x)
                    #print('after resnorm', x.shape)
                x = module(x)
                if j % 2 == 1:
                    x = x[:,:,:,:(-32 * 2**i)]
                    #print('after remove', x.shape)
                    if residual is not None:
                        x = x + residual
            skips.append(x)
            x = self.patchMerges[i](x)

         # ===== MIDDLE =====
        for j, module in enumerate(self.blockMiddle):
            if j % 2 == 0:
                residual = x if self.gradient_flowthrough[1] else None
                #print('middle before', x.shape, t3.shape)
                x = torch.concat((x, t3), dim=3)
                #print(x.shape, t3.shape)
                x = self.resnorms_middle[j // 2](x)
                #print(x.shape)
            x = module(x)
            if j % 2 == 1:
                x = x[:,:,:,:-128]
                if residual is not None:
                    x = x + residual

        # ===== UP =====
        for i, module_list in enumerate(self.blockUp):
            x = self.patchUnmerges[i](x)
            skip = skips[self.depth - i - 1]
            x = x + (self.skip_connects[i](skip) if self.skip_connect is not None else skip)

            for j, module in enumerate(module_list):
                if j % 2 == 0:
                    residual = x if self.gradient_flowthrough[2] else None
                    if i == 0:
                        #x = x + t4
                        x = torch.concat((x, t4), dim=3)
                    else:
                        #x = x + t5
                        x = torch.concat((x, t5), dim=3)
                    x = self.resnorms_up[i][j // 2](x)
                x = module(x)
                if j % 2 == 1:
                    x = x[:,:,:,:(-32 * 2**(self.depth - i - 1))]
                    if residual is not None:
                        x = x + residual
        
        x = self.embedding.decode(x, proj=True)
        x = x.permute(0,2,1,3,4).contiguous()
        return x

