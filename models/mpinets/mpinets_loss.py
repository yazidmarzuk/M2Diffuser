import torch
import torch.nn.functional as F

def point_clouds_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor, reduction: str="mean") -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The single loss value
    """
    return F.mse_loss(input_pc, target_pc, reduction=reduction) + \
            F.l1_loss(input_pc, target_pc, reduction=reduction)

def sdf_collision_loss(
    agent_pcs: torch.Tensor, 
    sdf_norm_values: torch.Tensor,
) -> torch.Tensor:
    """
    Calculating whether the robot (represented as a point cloud) is in collision 
    with any obstacles in the scene. 
    NOTE: Since our data is not collected in a structured environment like mpinets, 
    our method of calculating collision loss is different from mpinets, but our 
    method is still effective.

    Arguements:
        agent_pcs {torch.Tensor} -- The normalized point clouds for agent in the world frame.
        sdf_norm_values {torch.Tensor} -- The normalized scene SDF value in world frame, ranging in [-1, 1], 
                                        dimension like [B, grid_resolution, grid_resolution, grid_resolution].
    Returns:
        torch.Tensor -- Returns a collision loss using in model training.
    """
    agent_points_num = agent_pcs.shape[1]
    agent_sdf_batch = F.grid_sample(
        sdf_norm_values.unsqueeze(1), 
        agent_pcs[:,:,[2,1,0]].view(-1, agent_points_num, 1, 1, 3),
        padding_mode='border', 
        align_corners=False,
    )
    # if there are no penetrating vertices then set sdf_penetration_loss = 0
    if agent_sdf_batch.lt(0).sum().item() < 1:
        sdf_pene = torch.tensor(0.0, dtype=torch.float32, device=sdf_norm_values.device)
    else:
        sdf_pene = agent_sdf_batch[agent_sdf_batch < 0].abs().mean()
    return sdf_pene
