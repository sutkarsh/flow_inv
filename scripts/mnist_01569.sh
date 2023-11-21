python -m experiments.train_mario --ent-controller.aug-loss-factor-max 0.3 --ent-controller.aug-loss-factor-min 0.3 --model.n_copies 1 --optimizer.bs 512 --optimizer.lr 1e-3 --trainer.no_aug_until 3 --ent-controller.pid_warmup_epoch 3 --model.aug_type ROTATE_TRANS --trainer.dataset MNIST --optimizer.epochs 40 --ckpt_path ckpt/mnist_01569 --model.base-type CONDITIONAL_UNIFORM --model.loc_std 0.3 --model.n_mixtures 1 --trainer.dataset MNIST --trainer.start-scale 0.5 --trainer.mnist-classes 0 1 5 6 9