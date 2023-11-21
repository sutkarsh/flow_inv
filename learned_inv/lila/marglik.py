# From https://github.com/tychovdo/lila/blob/main/lila/marglik.py

# Taken from https://github.com/AlexImmer/Laplace/blob/main/laplace/marglik_training.py
# and modified for differentiable learning of invariances/augmentation strategies.
from copy import deepcopy
import logging
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
import tqdm

def valid_performance(model, test_loader, likelihood, method, device, augmenter=None):
    N = len(test_loader.dataset)
    perf = 0
    test_loader.transform = lambda x: x
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        if augmenter.n_samples <= 30: # Rejection sampling for small n_samples; can be ignored if n_samples is large enough
            with torch.no_grad():
                bignewn = 300 # Total pool of samples to draw and reject/accept from. This is a hacky shortcut for true rejection sampling, but it works well enough.
                newn = augmenter.n_samples
                x_aug = X
                input_embedding = augmenter.pose_net(x_aug)
                N_, C, H, W = x_aug.shape
                x_aug = x_aug[:, None].repeat(1, bignewn, 1, 1, 1).reshape(N_ * bignewn, C, H, W)
                input_embedding = input_embedding[:, None].repeat(1, bignewn, 1).reshape(N_ * bignewn, augmenter.pose_emb_dim)
                
                
                weights, logp = augmenter.nf_model.sample_weights(input_embedding)
                logp = logp.reshape(N_, bignewn)
                nf_dims = weights.shape[-1]
                weights = weights.reshape(N_, bignewn, nf_dims).detach()
                weights_norm = weights.norm(dim=-1)
                selected = (weights_norm < 1.0).float() # Reject samples that are too far away
                not_selected = (weights_norm >= 1.0).float()
                
                selected_numbered = torch.cumsum(selected, dim=1) * selected
                not_selected_numbered = (torch.cumsum(not_selected, dim=1) + selected_numbered.max(dim=1, keepdim=True).values) * not_selected
                all_numbered = selected_numbered + not_selected_numbered
                mask = (all_numbered <= newn).float() * (all_numbered >= 1).float()
                
                assert mask.sum(1).min() == mask.sum(1).max() == newn, f"Failed to sample good weights, mask: {mask.sum(1)}"

                # Select weights and logp based on mask
                weights = weights.reshape(N_ * bignewn, nf_dims)[mask.reshape(-1) > 0]
                logp = logp.reshape(N_ * bignewn)[mask.reshape(-1) > 0]
                
                x_aug = X[:, None].repeat(1, newn, 1, 1, 1).reshape(N_ * newn, C, H, W)
                weights = augmenter.nf_model.format_weights(weights)
                affine_matrices = augmenter.nf_model.weights_to_affine(weights)
                x_aug = augmenter.nf_model.apply_affine(x_aug, affine_matrices) 
                X = torch.cat([X[:, None, :, :, :], x_aug.reshape(N_, newn, C, H, W)], dim=1).reshape(N_, (newn + 1), C, H, W)
        else:
            X = augmenter(X)
            newn = augmenter.n_samples
        if method in ['avgfunc', 'augerino']:
            with torch.no_grad():
                slicesize = 10 # Batch up the data to avoid memory issues
                f = torch.cat([model(X[:, slicesize*i:slicesize*i+slicesize]) for i in range((newn//slicesize)+1)], dim=1).mean(dim=1)
            # f = model(X).mean(dim=1)
        else:
            f = model(X)
        if likelihood == 'classification':
            perf += (torch.argmax(f, dim=-1) == y).sum() / N
        else:
            perf += (f - y).square().sum() / N
    return perf.item()


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def grad_none_to_zero(e):
    return torch.zeros_like(e) if e.grad is None else e.grad

def gradient_to_vector(parameters):
    return parameters_to_vector([grad_none_to_zero(e) for e in parameters])


def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [grad_none_to_zero(e) for e in parameters])
