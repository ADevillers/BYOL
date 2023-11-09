import copy
import math
import numpy as np
import torch
import torchvision



class BYOL(torch.nn.Module):
    def __init__(self, resnet_type='resnet50', z_dim=256, cifar10=False):
        super().__init__()
        resnet_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152
        }

        self.resnet = resnet_dict[resnet_type](pretrained=False, zero_init_residual=True)
        self.h_dim = self.resnet.fc.in_features
        self.z_dim = z_dim

        self.resnet.fc = torch.nn.Identity()
        if cifar10:
            self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.resnet.maxpool = torch.nn.Identity()

        self.proj_head_inv = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, 4096, bias=False),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, self.z_dim)  # 256
        )

        self.predictor_inv = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 4096, bias=False),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, self.z_dim)  # 256
        )

    def forward(self, images):
        h = self.resnet(images)
        z = self.proj_head_inv(h)
        q = self.predictor_inv(z)

        return h, z, q



class M_BYOL(torch.nn.Module):
    def __init__(self, resnet, proj_head_inv, tau_base=0.996):
        super().__init__()

        self.tau_base = tau_base
        self.tau = self.tau_base

        self.m_resnet = copy.deepcopy(resnet)
        self.m_proj_head_inv = copy.deepcopy(proj_head_inv)

    def update(self, resnet, proj_head_inv, step, total_step):
        self.tau = 1. - (1. - self.tau_base)*(math.cos(np.pi*step/total_step) + 1.)/2.
        
        for current_params, m_params in zip(resnet.parameters(), self.m_resnet.parameters()):
            m_params.data = m_params.data*self.tau + current_params.data*(1. - self.tau)
        
        for current_params, m_params in zip(proj_head_inv.parameters(), self.m_proj_head_inv.parameters()):
            m_params.data = m_params.data*self.tau + current_params.data*(1. - self.tau)
    
    def forward(self, images):
        m_h = self.m_resnet(images)
        m_z = self.m_proj_head_inv(m_h)

        m_z = torch.cat([m_z[m_z.shape[0]//2:], m_z[:m_z.shape[0]//2]], dim=0)

        return m_h, m_z.detach()
