#CIFAR 10
python -m experiments.lila_exp  --dataset=cifar10 --augerino_reg=0.1 --lr=0.1 --lr_augerino=1e-4 --n_epochs=200 --stepLR=80 --stepLR_factor=0.1 --trans_scale=0.1 --loc_std 0.2 --name cifar10 --seed 0 --save_model ckpt/cifar10
#CIFAR 10 LT
python -m experiments.lila_exp  --dataset=cifar10lt --augerino_reg=0.05 --lr=0.1 --lr_augerino=1e-4 --n_epochs=100 --stepLR=80 --stepLR_factor=0.05 --trans_scale=0.5 --loc_std 0.5 --name "cifar10 lt" --seed 0 --save_model ckpt/cifar10lt
# FMNIST
python -m experiments.lila_exp  --dataset=fmnist --augerino_reg=0.03 --lr=0.03 --lr_augerino=1e-3 --n_epochs=200 --stepLR=80 --stepLR_factor=0.05 --trans_scale=0.5 --loc_std 0.5 --name "fmnist" --model cnn --seed 0 --save_model ckpt/fmnist
# MNIST
python -m experiments.lila_exp  --dataset=mnist --augerino_reg=0.01 --lr=1e-3 --lr_augerino=1e-4 --n_epochs=100 --stepLR=20 --stepLR_factor=0.1 --trans_scale=1 --loc_std 0.5 --name mnist --model cnn --n_epochs_burnin 1 --seed 0 --save_model ckpt/mnist


### 100 test time samples gets notably higher accuracy in most situations. It's no longer a fair comparison against the baselines, but it's worth a try:
#CIFAR 10
python -m experiments.lila_exp  --dataset=cifar10 --augerino_reg=0.1 --lr=0.1 --lr_augerino=1e-4 --n_epochs=200 --stepLR=80 --stepLR_factor=0.1 --trans_scale=0.1 --loc_std 0.2 --name "cifar10, 100 test samples" --seed 0 --test_n_samples 100
#CIFAR 10 LT
python -m experiments.lila_exp  --dataset=cifar10lt --augerino_reg=0.05 --lr=0.1 --lr_augerino=1e-4 --n_epochs=100 --stepLR=80 --stepLR_factor=0.05 --trans_scale=0.5 --loc_std 0.5 --name "cifar10 lt, 100 test samples" --seed 0 --test_n_samples 100
# FMNIST
python -m experiments.lila_exp  --dataset=fmnist --augerino_reg=0.03 --lr=0.03 --lr_augerino=1e-3 --n_epochs=200 --stepLR=80 --stepLR_factor=0.05 --trans_scale=0.5 --loc_std 0.5 --name "fmnist, 100 test samples" --model cnn --seed 0 --test_n_samples 100
# MNIST
python -m experiments.lila_exp  --dataset=mnist --augerino_reg=0.01 --lr=1e-3 --lr_augerino=1e-4 --n_epochs=100 --stepLR=20 --stepLR_factor=0.1 --trans_scale=1 --loc_std 0.5 --name "mnist, 100 test samples" --model cnn --n_epochs_burnin 1 --seed 0 --test_n_samples 100
