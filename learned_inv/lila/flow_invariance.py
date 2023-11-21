# From https://github.com/tychovdo/lila/blob/main/lila/augerino.py

import torch
from absl import logging
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim.adam import Adam
import torch.nn.functional as F
import tqdm

from learned_inv.lila.marglik import get_model_optimizer, get_scheduler, valid_performance
from learned_inv.utils import rot_img, rot_img_batch
import numpy as np
import os


def flow_invariance(model,
             train_loader,
             valid_loader=None,
             likelihood='classification',
             weight_decay=1e-4,
             aug_reg=0.01,
             n_epochs=500,
             lr=1e-3,
             lr_min=None,
             lr_aug=1e-3,
             optimizer='Adam',
             scheduler='exp',
             augmenter=None,
             use_logp=False,
             burnin=0,
             wandb=None,
             decay_epochs=None,
             stepLR=None,
             stepLR_factor=0.1,
             schedule_aug = False,
             test_n_samples=20,
             clip_grad=None):
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)

    end_point = decay_epochs if decay_epochs is not None else n_epochs

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr, weight_decay)
    if stepLR is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepLR*len(train_loader), gamma=stepLR_factor, last_epoch=-1)
        logging.info(f'Using StepLR with step_size={stepLR*len(train_loader)} and gamma={stepLR_factor}')
    else:
        scheduler = get_scheduler(scheduler, optimizer, train_loader, end_point, lr, lr_min)
        logging.info(f'Using Exponential Decay with lr_min={lr_min} and lr={lr} and end_point={end_point}')
        
    logging.info(f"FLOW_INVARIANCE with parameters: lr={lr}, lr_min={lr_min}, lr_aug={lr_aug}, weight_decay={weight_decay}, aug_reg={aug_reg}, n_epochs={n_epochs}, optimizer={optimizer}, scheduler={scheduler}, augmenter={augmenter}, use_logp={use_logp}, burnin={burnin}, wandb={wandb}, decay_epochs={decay_epochs}, stepLR={stepLR}, stepLR_factor={stepLR_factor}")

    logging.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimize_aug = (parameters_to_vector(augmenter.parameters()).requires_grad) and (aug_reg != 0.0)
    # set up augmentation optimizer
    if optimize_aug:
        logging.info('FLOW_INVARIANCE: optimize augmentation')
        aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug, weight_decay=0)

    if optimize_aug and (stepLR is not None) and schedule_aug:
        aug_sched = torch.optim.lr_scheduler.StepLR(aug_optimizer, step_size=stepLR*len(train_loader), gamma=stepLR_factor, last_epoch=-1)
        logging.info(f'Using StepLR with step_size={stepLR*len(train_loader)} and gamma={stepLR_factor} for augmentation')

    if likelihood == 'classification':
        criterion = CrossEntropyLoss()
    elif likelihood == 'regression':
        criterion = MSELoss()
    else:
        raise ValueError(f'Invalid likelihood: {likelihood}')

    losses = list()
    valid_perfs = list()
    step = 0

    train_loader.transform = lambda x: x
    for epoch in tqdm.trange(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # standard NN training per batch
        total = 0
        correct = 0
        for X, y in tqdm.tqdm(train_loader, leave=False, total=len(train_loader)):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            if optimize_aug:
                aug_optimizer.zero_grad()
            
            if epoch > burnin:
                X = augmenter(X)
            else:
                X = X[:, None]
            step += 1

            #! LILA seems to average before softmax, which seems different from our formulation and augerino's
            #! but honestly it doesn't make much of a difference.
            #! I'm following their convention here for consistency of results.

            f = model(X).mean(dim=1)
            loss = criterion(f, y).mean()
            
            correct += torch.sum(torch.argmax(f, dim=-1) == y).item()
            total += len(y)
            
            if optimize_aug and (epoch > burnin):
                loss += aug_reg * augmenter.logpmean * (epoch > burnin)
            loss.backward()
            optimizer.step() #!
            if optimize_aug and epoch > burnin:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(augmenter.parameters(), clip_grad)
                aug_optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()
            if optimize_aug and (stepLR is not None) and schedule_aug:
                aug_sched.step()

        losses.append(epoch_loss)
        # compute validation error to report during training
        if valid_loader is not None:
            assert hasattr(augmenter, 'pose_net'), "This implementation is only guaranteed to work for our method"
            if hasattr(augmenter, 'n_samples'):
                n_samps = augmenter.n_samples
                augmenter.n_samples = test_n_samples # Temporarily set to test_n_samples, only for validation
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, method='avgfunc', device=device, augmenter=augmenter) 
                valid_perfs.append(valid_perf)
                logging.info(f'FLOW_INVARIANCE[epoch={epoch}]: train perf {(correct/total)*100:.2f}% validation performance {valid_perf*100:.2f}.%')
                wandb.log({'train_perf': (correct/total)*100, 'valid_perf': valid_perf*100}, step=epoch)
            if hasattr(augmenter, 'n_samples') and hasattr(augmenter, 'pose_net'):
                augmenter.n_samples = n_samps
    if optimize_aug:
        return model, losses, valid_perfs, None
    return model, losses, valid_perfs, None