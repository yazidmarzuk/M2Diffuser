from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from env.agent.mec_kinova import MecKinova

from models.optimizer.optimizer import Optimizer
from models.base import PLANNER

@PLANNER.register()
class GreedyMKPlanner(Optimizer):

    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        self.scale = cfg.scale

        self.greedy_type = cfg.greedy_type

        self.clip_grad_by_value = cfg.clip_grad_by_value

        self.greedy_weight_base_arm = cfg.greedy_weight_base_arm

    def objective(self, x: torch.Tensor, data: Dict):
        """ Compute gradient for planner guidance

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data

        Return:
            The optimizer objective value of current step
        """
        loss = 0.

        ## important!!! scale x with region_area
        ## convert x to reality psoition
        start = data['start']
        target = data['target']  # <B, 7>

        if self.greedy_type == 'last_frame':
            # loss += F.l1_loss(x[:, -1, :], target, reduction='mean')
            base_loss = F.l1_loss(x[:, -1, :MecKinova.BASE_DOF], target[:, :MecKinova.BASE_DOF], reduction='sum') / MecKinova.DOF
            arm_loss = F.l1_loss(x[:, -1, MecKinova.BASE_DOF:], target[:, MecKinova.BASE_DOF:], reduction='sum') / MecKinova.DOF
            loss += self.greedy_weight_base_arm  * base_loss + arm_loss
        elif self.greedy_type == 'all_frame':
            # loss += F.l1_loss(x, target.unsqueeze(1), reduction='mean')
            base_loss = F.l1_loss(x[:, :, :MecKinova.BASE_DOF], target.unsqueeze(1)[:, :, :MecKinova.BASE_DOF], reduction='sum') / MecKinova.DOF
            arm_loss = F.l1_loss(x[:, :, MecKinova.BASE_DOF:], target.unsqueeze(1)[:, :, MecKinova.BASE_DOF:], reduction='sum') / MecKinova.DOF
            loss += self.greedy_weight_base_arm  * base_loss + arm_loss
        elif self.greedy_type == 'all_frame_exp':
            traj_dist = torch.norm(x - target.unsqueeze(1), dim=-1, p=1)
            loss += (-1.0) * torch.exp(1 / traj_dist.clamp(min=0.01)).sum()
        else:
            raise Exception('Unsupported greedy type')
        
        #! 同时限定了初始的连续性
        loss += F.l1_loss(x[:, 0, :], start.squeeze(1), reduction='mean')

        return (-1.0) * loss

    def gradient(self, x: torch.Tensor, data: Dict, t: int, timesteps: int, variance: torch.Tensor, fitting: bool=False) -> torch.Tensor:
        """ Compute gradient for planner guidance
        Args:
            x: the denosied signal at current step
            data: data dict that provides original data

        Return:
            Commputed gradient
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.objective(x_in, data)
            grad = torch.autograd.grad(obj, x_in)[0]

            # print(f'obj: {-obj.detach().cpu()}')

            ## clip gradient by value
            grad = grad * self.scale
            grad = torch.clip(grad, **self.clip_grad_by_value)
            ## TODO clip gradient by norm

            return grad