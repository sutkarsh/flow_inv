import torch
import torch.nn.functional as F
import numpy as np
from math import pi
import learned_inv.aug as aug
from learned_inv.model import SimpleConv
from learned_inv.aug import NFAug
import normflows as nf
import torch.nn as nn
import logging



class NFAugLayer(nn.Module):
    def __init__(self, augtype='rot', mp=4, trans_scale=0.1, base_type='gauss', inp_channels=3, n_mixtures=100,
                 ignore_intermediate_tanh=False, num_layers = 0, hard_gs = True, loc_std = 0.5, logp_sq = False,
                 gumbel_tau = 0.1, tanh_width = 1.5, n_samples = 32, padding_mode='zeros', *args, **kwargs):
        super(NFAugLayer, self).__init__()

        self.aug_type = {'rot': aug.AugType.ROTATE,
                         'trans': aug.AugType.TRANS,
                         'rot_trans': aug.AugType.ROTATE_TRANS,
                         'full': aug.AugType.FULL}[augtype]
        
        
        self.base_type = {'gauss': aug.BaseType.CONDITIONAL_GAUSSIAN_MIX}[base_type]

        pose_emb_dim = 32
        pose_emb_netwidth = 8
        in_channel = inp_channels
        self.trans_scale = trans_scale
        self.n_samples = n_samples
        
        self.pose_net = SimpleConv(c=pose_emb_netwidth, num_classes=pose_emb_dim, in_channel=in_channel, mp=mp).cuda()
        pose_net_param_count = sum(p.numel() for p in self.pose_net.parameters() if p.requires_grad)
        print(f"Pose net param count: {pose_net_param_count/1e6:.2f}M")
        self.gating = False # Ignore this
        
        self.pose_emb_dim = pose_emb_dim
        assert self.base_type == aug.BaseType.CONDITIONAL_GAUSSIAN_MIX, "Only Gaussian mixtures supported"

        logging.info(f"Using NF Augmentation Layer with type: {self.aug_type}")
        self.nf_model = NFAug(trans_scale=trans_scale, 
                              pose_emb_dim=self.pose_emb_dim,
                              aug_type=self.aug_type,
                              base_type=self.base_type,
                              n_mixtures=n_mixtures,
                              num_layers=num_layers,
                              hard_gs=hard_gs,
                              loc_std=loc_std,
                              gumbel_tau=gumbel_tau,
                              tanh_width=tanh_width,
                              logp_sq=logp_sq,
                              ignore_intermediate_tanh=ignore_intermediate_tanh,
                              padding_mode=padding_mode,
                              )
        print(f"NF model param count: {sum(p.numel() for p in self.nf_model.parameters() if p.requires_grad)}")
    def forward(self, x):
        input_embedding = self.pose_net(x)

        N, C, H, W = x.shape
        x = x[:, None].repeat(1, self.n_samples, 1, 1, 1).reshape(N * self.n_samples, C, H, W)
        input_embedding = input_embedding[:, None].repeat(1, self.n_samples, 1).reshape(N * self.n_samples, self.pose_emb_dim)
        
        x, logp = self.nf_model(x, input_embedding)
        x = x.reshape(N, self.n_samples, C, H, W)
        self.logpmean = logp.mean()
        return x
