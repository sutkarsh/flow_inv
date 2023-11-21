python -m experiments.train_mario --model.n_copies 1 --optimizer.bs 256 --trainer.no_aug_until 3 --ent-controller.pid_warmup_epoch 3 --model.aug_type ROTATE_TRANS --trainer.dataset MARIO --optimizer.epochs 150 --trainer.rot-range-factor 6 --trainer.n_modes 3 --model.base_type CONDITIONAL_GAUSSIAN_MIX --trainer.freeze-classifier --model.n_mixtures 90 --optimizer.lr 1e-3 --model.loc_std 0.7 --ent-controller.aug-loss-factor-max 0.5 --ent-controller.aug-loss-factor-min 0.5 --ckpt_path ckpt/3mode --ent-controller.ent-min 0 --ent-controller.ent-max 6  --model.gumbel-tau 0.1 # --model.hard_gs --model.gumbel-tau