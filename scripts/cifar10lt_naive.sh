# CIFAR 10 LT without any augmentations (naive); only needed to make the EKLD plots
python -m experiments.lila_exp  --dataset=cifar10lt --name "cifar10 lt naive" --seed 0 --n_epochs_burnin 100  --augerino_reg=0.05 --lr=0.1 --lr_augerino=1e-4 --n_epochs=100 --stepLR=80 --stepLR_factor=0.05 --trans_scale=0.5 --loc_std 0.5 --save_model ckpt/cifar10lt_naive
