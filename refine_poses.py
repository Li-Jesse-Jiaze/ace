# Copyright © Niantic, Inc. 2022.

import logging

import torch
from torch import optim

import roma

_logger = logging.getLogger(__name__)

class LMOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, lambda_init=1e-3, lambda_increase=10, lambda_decrease=10, max_iter=20):
        """
        Levenberg-Marquardt Optimizer.

        Args:
            params (iterable): Parameters to optimize.
            lr (float): Learning rate scaling factor.
            lambda_init (float): Initial damping factor.
            lambda_increase (float): Factor to increase lambda when loss increases.
            lambda_decrease (float): Factor to decrease lambda when loss decreases.
            max_iter (int): Maximum iterations per step (not used here but kept for compatibility).
        """
        defaults = dict(lr=lr, lambda_=lambda_init, lambda_increase=lambda_increase, 
                        lambda_decrease=lambda_decrease, max_iter=max_iter)
        super(LMOptimizer, self).__init__(params, defaults)

    def step(self, loss):
        """
        Performs a single optimization step.

        Args:
            loss (torch.Tensor): The current loss tensor.
        """
        loss.backward()

        for group in self.param_groups:
            lambda_ = group['lambda_']
            lr = group['lr']

            params = group['params']
            # Collect gradients and flatten them
            grads = []
            for p in params:
                if p.grad is None:
                    grads.append(torch.zeros_like(p))
                else:
                    grads.append(p.grad.flatten())
            grads = torch.cat(grads) * 1e-2  # (num_params,)

            # Compute JTJ (approximate Hessian)
            JTJ = torch.ger(grads, grads)  # Outer product (num_params, num_params)

            # Add damping factor to the diagonal
            JTJ += lambda_ * torch.eye(JTJ.size(0), device=JTJ.device)

            # Compute the parameter update
            try:
                delta = -torch.linalg.solve(JTJ, grads)
            except RuntimeError:
                # In case JTJ is singular, use pseudo-inverse
                delta = -torch.matmul(torch.pinverse(JTJ), grads)

            # Scale delta by learning rate
            delta = delta * lr

            # Apply the update to parameters
            with torch.no_grad():
                idx = 0
                for p in params:
                    numel = p.numel()
                    if p.grad is not None:
                        delta_p = delta[idx:idx+numel].view_as(p)
                        p.add_(delta_p)
                    idx += numel


def se3_exp(xi):
    """
    Exponential map from se(3) to SE(3)
    xi: (N,6) tensor, where xi[:, :3] is omega, xi[:, 3:] is v
    Returns: (N,4,4) SE(3) matrices
    """
    omega = xi[:, :3]  # (N,3)
    v = xi[:, 3:]      # (N,3)
    R = roma.rotvec_to_rotmat(omega)
    # Construct SE(3) matrices
    SE3 = torch.zeros(xi.shape[0], 4, 4).to(xi.device)
    SE3[:, :3, :3] = R
    SE3[:, :3, 3] = v
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
    omega = roma.rotmat_to_rotvec(R)
    xi = torch.cat([omega, t], dim=1)  # (N,6)

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
        self.max_iter = options.iterations - options.pose_refinement_wait

        # pose buffer for current estimate of refined poses
        self.pose_buffer = None
        # pose buffer for original poses (using Lie group SE(3))
        self.pose_buffer_orig = None
        # network predicting pose updates (depending on the optimization strategy)
        self.pose_network = None
        # optimizer for pose updates
        self.pose_optimizer = None

    def create_pose_buffer(self):
        """
        Populate internal pose buffers and set up the pose optimization strategy.
        """
        if self.refinement_strategy == 'mlp':
            raise NotImplementedError("MLP refinement strategy is not implemented in this modification.")

        # Initialize pose_buffer_orig as SE(3) matrices
        self.pose_buffer_orig = torch.zeros(len(self.dataset), 4, 4).to(self.device)

        for pose_idx, pose in enumerate(self.dataset.poses):
            pose_matrix = pose.inverse().clone()  # (4, 4)
            self.pose_buffer_orig[pose_idx] = pose_matrix

        if self.refinement_strategy == 'none':
            # No optimization needed; poses remain as original
            return

        elif self.refinement_strategy == 'naive':
            # Initialize delta pose_buffer as zeros (no change)
            self.pose_buffer = torch.zeros(len(self.dataset), 6, device=self.device, requires_grad=True)
            # Set up LM optimizer to optimize delta poses
            self.pose_optimizer = LMOptimizer([self.pose_buffer], lr=self.learning_rate, max_iter=self.max_iter)

    def get_all_original_poses(self):
        """
        Get all original poses as SE(3) matrices.
        """
        return self.pose_buffer_orig.clone()

    def get_all_current_poses(self):
        """
        Get all current estimates of refined poses.
        """
        if self.refinement_strategy == 'none':
            # Just return original poses
            return self.get_all_original_poses()
        elif self.refinement_strategy == 'naive':
            # Compute refined poses: delta_SE3 * original_pose
            delta_SE3 = se3_exp(self.pose_buffer)  # (N,4,4)
            refined_SE3 = torch.bmm(delta_SE3, self.pose_buffer_orig)  # (N,4,4)
            return refined_SE3

    def get_current_poses(self, original_poses_b44, original_poses_indices):
        """
        Get current estimates of refined poses for a subset of the original poses.

        @param original_poses_b44: original poses, shape (b, 4, 4)
        @param original_poses_indices: indices of the original poses in the dataset
        """
        if self.refinement_strategy == 'none':
            # Just return original poses
            return original_poses_b44.clone()
        elif self.refinement_strategy == 'naive':
            # Get delta_xi for the specified poses
            delta_xi = self.pose_buffer[original_poses_indices.view(-1)].clone()  # (b,6)
            # Compute delta SE(3)
            delta_SE3 = se3_exp(delta_xi)  # (b,4,4)
            # Compute refined poses: delta_SE3 * original_pose
            refined_SE3 = torch.bmm(delta_SE3, original_poses_b44)  # (b,4,4)
            return refined_SE3

    def zero_grad(self, set_to_none=False):
        if self.pose_optimizer is not None:
            # Manually zero the gradients of pose_buffer
            self.pose_buffer.grad.zero_() if self.pose_buffer.grad is not None else None

    def step(self, loss):
        """
        Performs a single optimization step.

        Args:
            loss (torch.Tensor): The current loss tensor.
        """
        if self.pose_optimizer is not None:
            self.pose_optimizer.step(loss)
