"""Landmark augmentations used by MSLR (training samples only)."""
import random

import torch


def augment_landmarks(x, rotation_p=0.30, translation_flip_p=0.50, fusion_p=0.20):
    """Augment [T,276] from this project's 258 channels plus 18 zero pads."""
    x = x.clone()
    hand_xyz = x[:, :126].reshape(x.shape[0], 42, 3)
    pose = x[:, 126:258].reshape(x.shape[0], 33, 4)
    pose_xyz = pose[..., :3]
    if random.random() < rotation_p:
        angle = random.uniform(-30.0, 30.0) * 3.141592653589793 / 180.0
        c, s = torch.cos(torch.tensor(angle, device=x.device, dtype=x.dtype)), torch.sin(torch.tensor(angle, device=x.device, dtype=x.dtype))
        rot = torch.stack([torch.stack([c, -s, c * 0]), torch.stack([s, c, c * 0]),
                           torch.stack([c * 0, c * 0, c * 0 + 1])])
        hand_xyz = hand_xyz @ rot.T
        pose_xyz = pose_xyz @ rot.T
    if random.random() < translation_flip_p:
        shift = torch.empty((1, 1, 3), device=x.device, dtype=x.dtype).uniform_(-0.05, 0.05)
        hand_xyz = hand_xyz + shift
        pose_xyz = pose_xyz + shift
        hand_xyz[..., 0] = -hand_xyz[..., 0]
        pose_xyz[..., 0] = -pose_xyz[..., 0]
    x[:, :126] = hand_xyz.reshape(x.shape[0], 126)
    pose[..., :3] = pose_xyz
    x[:, 126:258] = pose.reshape(x.shape[0], 132)
    if x.shape[0] > 1 and random.random() < fusion_p:
        weight = random.uniform(0.25, 0.75)
        x[1:] = weight * x[1:] + (1.0 - weight) * x[:-1].clone()
    return x
