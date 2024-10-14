# Copyright © Niantic, Inc. 2022.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import roma

_logger = logging.getLogger(__name__)


class PoseNetwork(nn.Module):
    """
    MLP network predicting a pose update.
    It takes 12 inputs (3x4 pose) and predicts 12 values, e.g. used as additive offsets.
    """

    def __init__(self, num_head_blocks, channels=512):
        super(PoseNetwork, self).__init__()

        self.in_channels = 12
        self.head_channels = channels  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(PoseNetwork, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(PoseNetwork, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(PoseNetwork, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc3 = nn.Conv2d(self.head_channels, 12, 1, 1, 0)

    def forward(self, res):

        x = F.relu(self.conv1(res))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        pose_update = F.relu(self.fc1(res))
        pose_update = F.relu(self.fc2(pose_update))
        pose_update = self.fc3(pose_update)

        return pose_update


def skew_symmetric(omega):
    """
    Compute skew-symmetric matrix of omega
    omega: (N,3)
    Returns: (N,3,3)
    """
    N = omega.shape[0]
    zero = torch.zeros(N, 1).to(omega.device)
    omega_x = omega[:, 0].unsqueeze(1)
    omega_y = omega[:, 1].unsqueeze(1)
    omega_z = omega[:, 2].unsqueeze(1)

    row0 = torch.cat([zero, -omega_z, omega_y], dim=1)
    row1 = torch.cat([omega_z, zero, -omega_x], dim=1)
    row2 = torch.cat([-omega_y, omega_x, zero], dim=1)

    skew = torch.stack([row0, row1, row2], dim=1)  # (N,3,3)

    return skew


def se3_exp(xi):
    """
    Exponential map from se(3) to SE(3)
    xi: (N,6) tensor, where xi[:, :3] is omega, xi[:, 3:] is v
    Returns: (N,4,4) SE(3) matrices
    """
    omega = xi[:, :3]  # (N,3)
    v = xi[:, 3:]      # (N,3)

    theta = omega.norm(dim=1, keepdim=True)  # (N,1)
    epsilon = 1e-8
    theta = theta + epsilon

    omega_hat = skew_symmetric(omega)  # (N,3,3)

    # Rodrigues' formula for rotation matrix
    A = torch.sin(theta) / theta       # (N,1)
    B = (1 - torch.cos(theta)) / (theta ** 2)  # (N,1)
    C = (1 - A) / (theta ** 2)         # (N,1)

    E = torch.eye(3).unsqueeze(0).to(xi.device)  # (1,3,3)

    R = E + A.view(-1,1,1) * omega_hat + B.view(-1,1,1) * torch.bmm(omega_hat, omega_hat)  # (N,3,3)
    V = E + B.view(-1,1,1) * omega_hat + C.view(-1,1,1) * torch.bmm(omega_hat, omega_hat)  # (N,3,3)

    t = torch.bmm(V, v.unsqueeze(-1)).squeeze(-1)  # (N,3)

    # Construct SE(3) matrices
    SE3 = torch.zeros(xi.shape[0], 4, 4).to(xi.device)
    SE3[:, :3, :3] = R
    SE3[:, :3, 3] = t
    SE3[:, 3, 3] = 1.0

    return SE3


def se3_log(SE3):
    """
    Logarithm map from SE(3) to se(3)
    SE3: (N,4,4) SE(3) matrices
    Returns: (N,6) xi vectors
    """
    R = SE3[:, :3, :3]  # (N,3,3)
    t = SE3[:, :3, 3]   # (N,3)

    omega = roma.mappings.rotmat_to_rotvec(R)  # (N,3)
    theta = omega.norm(dim=1, keepdim=True)  # (N,1)
    epsilon = 1e-8
    theta = theta + epsilon

    omega_hat = skew_symmetric(omega)  # (N,3,3)

    # Compute V_inv
    half_theta = 0.5 * theta
    cot_half_theta = 1.0 / torch.tan(half_theta)
    V_inv = (torch.eye(3).to(SE3.device).unsqueeze(0) - 0.5 * omega_hat + (1.0 / theta**2) * (1 - (theta * cot_half_theta) / 2) * torch.bmm(omega_hat, omega_hat))

    v = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)  # (N,3)

    xi = torch.cat([omega, v], dim=1)  # (N,6)

    return xi


class PoseRefiner:
    """
    Handles refinement of per-image pose information during ACE training.

    Support three variants.
    1. 'none': no pose refinement
    2. 'naive': back-prop to poses directly using Lie groups and Lie algebras
    3. 'mlp': use a network to predict pose updates
    """

    def __init__(self, dataset, device, options):

        self.dataset = dataset
        self.device = device

        # set refinement strategy
        if options.pose_refinement not in ['none', 'naive', 'mlp']:
            raise ValueError(f"Pose refinement strategy {options.pose_refinement} not supported")
        self.refinement_strategy = options.pose_refinement

        # set options
        self.learning_rate = options.pose_refinement_lr
        self.update_weight = options.pose_refinement_weight
        self.orthonormalization = options.refinement_ortho

        # pose buffer for current estimate of refined poses
        self.pose_buffer = None
        # pose buffer for original poses
        self.pose_buffer_orig = None
        # network predicting pose updates (depending on the optimization strategy)
        self.pose_network = None
        # optimizer for pose updates
        self.pose_optimizer = None

    def create_pose_buffer(self):
        """
        Populate internal pose buffers and set up the pose optimization strategy.
        """
        self.pose_buffer_orig = torch.zeros(len(self.dataset), 6)

        # fill pose buffer with Lie algebra elements (omega and translation)
        for pose_idx, pose in enumerate(self.dataset.poses):
            # Convert pose to 4x4 matrix
            pose_matrix = torch.eye(4)
            pose_matrix[:3, :4] = pose.inverse().clone()[:3, :4]  # (4, 4)

            # Compute logarithm map of SE(3) to get xi (omega and v)
            xi = se3_log(pose_matrix.unsqueeze(0))  # (1,6)
            self.pose_buffer_orig[pose_idx] = xi.squeeze()

        self.pose_buffer = self.pose_buffer_orig.contiguous().to(self.device, non_blocking=True)

        # set the pose optimization strategy
        if self.refinement_strategy == 'none':
            # will keep original poses
            pass
        elif self.refinement_strategy == 'naive':
            # back-prop to pose parameters (Lie algebra elements) directly
            self.pose_buffer = self.pose_buffer.detach().requires_grad_()
            self.pose_optimizer = optim.AdamW([self.pose_buffer], lr=self.learning_rate)
        else:
            # use small network to predict pose updates
            self.pose_network = PoseNetwork(0, 128)
            self.pose_network = self.pose_network.to(self.device)
            self.pose_network.train()
            self.pose_optimizer = optim.AdamW(self.pose_network.parameters(), lr=self.learning_rate)

    def get_all_original_poses(self):
        """
        Get all original poses.
        """
        # Convert Lie algebra elements back to SE(3) matrices
        poses_b44 = se3_exp(self.pose_buffer_orig.to(self.device))
        return poses_b44

    def get_all_current_poses(self):
        """
        Get all current estimates of refined poses.
        """
        if self.refinement_strategy == 'none':
            # just return original poses
            return self.get_all_original_poses()
        elif self.refinement_strategy == 'naive':
            # return current state of the pose buffer
            current_params = self.pose_buffer.clone()  # (N,6)

            # Compute SE(3) matrices by exponential map
            current_SE3_N44 = se3_exp(current_params)  # (N,4,4)

            return current_SE3_N44
        else:
            # predict pose updates with current state of the network
            with torch.no_grad():
                # return current state of the pose buffer
                output_params = self.pose_buffer.clone()

                # predict current poses
                # Note: For 'mlp' strategy, we need to modify this part accordingly
                output_poses = self.pose_buffer_orig.to(self.device)
                predicted_updates = self.pose_network(output_poses.view(-1, 12, 1, 1))
                updated_params = output_params + self.update_weight * predicted_updates.view(-1, 6)
                current_SE3_N44 = se3_exp(updated_params)

                return current_SE3_N44

    def get_current_poses(self, original_poses_b44, original_poses_indices):
        """
        Get current estimates of refined poses for a subset of the original poses.

        @param original_poses_b44: original poses, shape (b, 4, 4)
        @param original_poses_indices: indices of the original poses in the dataset
        """
        if self.refinement_strategy == 'none':
            # just return original poses
            return original_poses_b44.clone()
        elif self.refinement_strategy == 'naive':
            # get current state of the pose parameters (Lie algebra elements) from buffer
            current_params_b6 = self.pose_buffer[original_poses_indices].squeeze()  # (b, 6)

            # Compute SE(3) matrices by exponential map
            current_SE3_b44 = se3_exp(current_params_b6)  # (b,4,4)

            return current_SE3_b44
        else:
            # predict pose updates with current state of the network
            output_poses = self.pose_buffer_orig[original_poses_indices].to(self.device)
            predicted_updates = self.pose_network(output_poses.view(-1, 12, 1, 1))
            updated_params = output_poses + self.update_weight * predicted_updates.view(-1, 6)
            current_SE3_b44 = se3_exp(updated_params)

            return current_SE3_b44

    def zero_grad(self, set_to_none=False):
        if self.pose_optimizer is not None:
            self.pose_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.pose_optimizer is not None:
            self.pose_optimizer.step()