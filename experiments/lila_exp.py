# Code from LILA, modified to run with our methods

import numpy as np
import torch
import pickle
import os
from copy import deepcopy
from absl import app, flags, logging
from pathlib import Path
from torchvision import transforms
from torch.nn.utils.convert_parameters import parameters_to_vector
import coloredlogs
import wandb

coloredlogs.install(level='INFO')

from learned_inv.lila.datasets import RotatedMNIST, TranslatedMNIST, ScaledMNIST 
from learned_inv.lila.datasets import RotatedFashionMNIST, TranslatedFashionMNIST, ScaledFashionMNIST
from learned_inv.lila.datasets import RotatedCIFAR10, TranslatedCIFAR10, ScaledCIFAR10, CIFAR10LT
from learned_inv.lila.utils import TensorDataLoader, dataset_to_tensors, set_seed
from learned_inv.lila.models import MLP, LeNet, ResNet, WideResNet, layer13s
from learned_inv.lila.layers import NFAugLayer
from learned_inv.lila.flow_invariance import flow_invariance
from torch.utils.data import DataLoader


FLAGS = flags.FLAGS
np.set_printoptions(precision=3)

flags.DEFINE_integer('seed', 137, 'Random seed for data generation and model initialization.')
flags.DEFINE_enum(
    'method', 'augerino', ['avgfunc', 'augerino'],
    'Available methods: `avgfunc` is averaging functions, `augerino` is by Benton et al.')
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['mnist', 'mnist_r90', 'mnist_r180', 'translated_mnist', 'scaled_mnist', 
     'fmnist', 'fmnist_r90', 'fmnist_r180', 'translated_fmnist', 'scaled_fmnist', 
     'cifar10',  'cifar10lt', 'cifar10_r90', 'cifar10_r180', 'translated_cifar10', 'scaled_cifar10'],
    'Available methods: `mnist` is plain MNIST data, `mnist_r90` is partially-rotated ±90° MNIST, `mnist_r180` is fully-rotated ±180° MNIST')
flags.DEFINE_enum('model', 'resnet_8_16', ['mlp', 'cnn', 'resnet_8_16', 'resnet_8_8', 'wrn', 'layer13s'], help='model architecture')

flags.DEFINE_string('name', 'test', 'experiment name')
flags.DEFINE_bool('use_logp', True, 'whether to use logp') # True
flags.DEFINE_bool('conditional', False, 'conditional') # True
flags.DEFINE_bool('NF', True, 'use NFAug') # True
flags.DEFINE_bool('flip', False, 'flip images') # True

flags.DEFINE_float('augerino_reg', 1e-2, 'Augerino regularization strength (default from paper).') # [0.01,0.1,1]
flags.DEFINE_float('clip_grad', None, 'Augmenter gradient clipping')
flags.DEFINE_string('augtype', 'full', 'type of augmentation') # [rot_trans, full]
flags.DEFINE_float('trans_scale', 0.1, 'translation scale for our method') # [0.1, 1]
flags.DEFINE_float('lr_augerino', 0.005, 'lr')  # [0.005, 0.001, 0.0001]
flags.DEFINE_integer('n_mixtures', 10, 'Number of mixtures in the distribution') # [1, 2, 3]
flags.DEFINE_integer('decay_epochs', None, 'decay to minimum LR by this point')
flags.DEFINE_integer('stepLR', None, 'stepLR gap (default is None, i.e. no stepLR')
flags.DEFINE_float('stepLR_factor', 0.1, 'how much to decay after each step')  # [0.005, 0.001, 0.0001]
flags.DEFINE_enum('padding_mode', 'zeros', ['zeros', 'border', 'reflection'], help='Type of padding for augmentation',)

flags.DEFINE_integer('n_epochs', 200, 'Number of epochs')
flags.DEFINE_integer('batch_size', 256, 'Batch size for stochastic estimates. If set to -1 (default), use full batch.')
flags.DEFINE_integer('valid_batch_size', 64, 'Batch size for stochastic estimates. If set to -1 (default), use full batch.')
flags.DEFINE_integer('subset_size', 50000, 'Size of random subset, subset_size <= 0 means full data set')
flags.DEFINE_integer('n_samples_aug', 20, 'number of augmentation samples if applicable')
flags.DEFINE_integer('test_n_samples', 20, '(test-only) number of augmentation samples if applicable')
flags.DEFINE_bool('save', True, 'Whether to save the experiment outcome as pickle')
flags.DEFINE_string('save_model', None, 'Whether to save the model and where to save it')
flags.DEFINE_enum('device', 'cuda', ['cpu', 'cuda'], 'Torch device')
flags.DEFINE_bool('download', False, 'whether to (re-)download data set')
flags.DEFINE_string('data_root', 'tmp', 'root of the data directory')
flags.DEFINE_bool('schedule_aug', False, 'whether to decay aug LR along with the main LR')
flags.DEFINE_integer('num_flow_layers', 12, 'Number of flow layers')
flags.DEFINE_float('tanh_width', 1, 'tanh width')  # [0.005, 0.001, 0.0001]
flags.DEFINE_float('loc_std', 0.5, 'loc_std')  # [0.005, 0.001, 0.0001]

flags.DEFINE_float('lr', 0.1, 'lr')
flags.DEFINE_float('lr_min', 1e-6, 'decay target of the learning rate')
flags.DEFINE_float('lr_hyp', 0.05, 'lr hyper')
flags.DEFINE_integer('n_epochs_burnin', 10, 'number of epochs without marglik opt')

def main(argv):
    # dataset-specific static transforms (preprocessing)    
    wandb.init(project="lila_flow_test" , config=FLAGS.flag_values_dict(), name=FLAGS.name)
    wandb.save(str(FLAGS))
    wandb.run.log_code(".")

    if 'mnist' in FLAGS.dataset:
        transform = transforms.ToTensor()
    elif 'cifar' in FLAGS.dataset:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        if FLAGS.flip:
            logging.info('Using random horizontal flip.')
            transform_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]
        else:
            transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        transform = transforms.Compose(transform_list)
    else:
        raise NotImplementedError(f'Transform for {FLAGS.dataset} unavailable.')

    # Load data
    if FLAGS.dataset == 'mnist':
        train_dataset = RotatedMNIST(FLAGS.data_root, 0, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedMNIST(FLAGS.data_root, 0, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'mnist_r90':
        train_dataset = RotatedMNIST(FLAGS.data_root, 90, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedMNIST(FLAGS.data_root, 90, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'mnist_r180':
        train_dataset = RotatedMNIST(FLAGS.data_root, 180, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedMNIST(FLAGS.data_root, 180, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'translated_mnist':
        train_dataset = TranslatedMNIST(FLAGS.data_root, 8, train=True, download=FLAGS.download, transform=transform)
        test_dataset = TranslatedMNIST(FLAGS.data_root, 8, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_mnist':
        train_dataset = ScaledMNIST(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledMNIST(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'fmnist':
        train_dataset = RotatedFashionMNIST(FLAGS.data_root, 0, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedFashionMNIST(FLAGS.data_root, 0, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'fmnist_r90':
        train_dataset = RotatedFashionMNIST(FLAGS.data_root, 90, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedFashionMNIST(FLAGS.data_root, 90, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'fmnist_r180':
        train_dataset = RotatedFashionMNIST(FLAGS.data_root, 180, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedFashionMNIST(FLAGS.data_root, 180, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'translated_fmnist':
        train_dataset = TranslatedFashionMNIST(FLAGS.data_root, 8, train=True, download=FLAGS.download, transform=transform)
        test_dataset = TranslatedFashionMNIST(FLAGS.data_root, 8, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_fmnist':
        train_dataset = ScaledFashionMNIST(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledFashionMNIST(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10lt':
        train_dataset = CIFAR10LT(FLAGS.data_root, train=True, download=FLAGS.download, transform=transform)
        test_dataset = CIFAR10LT(FLAGS.data_root, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10':
        train_dataset = RotatedCIFAR10(FLAGS.data_root, 0, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedCIFAR10(FLAGS.data_root, 0, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10_r90':
        train_dataset = RotatedCIFAR10(FLAGS.data_root, 90, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedCIFAR10(FLAGS.data_root, 90, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'cifar10_r180':
        train_dataset = RotatedCIFAR10(FLAGS.data_root, 180, train=True, download=FLAGS.download, transform=transform)
        test_dataset = RotatedCIFAR10(FLAGS.data_root, 180, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'translated_cifar10':
        train_dataset = TranslatedCIFAR10(FLAGS.data_root, 8, train=True, download=FLAGS.download, transform=transform)
        test_dataset = TranslatedCIFAR10(FLAGS.data_root, 8, train=False, download=FLAGS.download, transform=transform)
    elif FLAGS.dataset == 'scaled_cifar10':
        train_dataset = ScaledCIFAR10(FLAGS.data_root, np.log(2), train=True, download=FLAGS.download, transform=transform)
        test_dataset = ScaledCIFAR10(FLAGS.data_root, np.log(2), train=False, download=FLAGS.download, transform=transform)
    else:
        raise NotImplementedError(f'Unknown dataset: {FLAGS.dataset}')

    set_seed(FLAGS.seed)

    # Subset the data if subset_size is given.
    subset_size = len(train_dataset) if FLAGS.subset_size <= 0 else FLAGS.subset_size
    if subset_size < len(train_dataset):
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    else:
        subset_indices = None
    X_train, y_train = dataset_to_tensors(train_dataset, subset_indices, FLAGS.device)
    X_test, y_test = dataset_to_tensors(test_dataset, None, FLAGS.device)


    mp = 2 if 'mnist' in FLAGS.dataset else 4
    inp_channels = 1 if 'mnist' in FLAGS.dataset else 3
    augmenter = NFAugLayer(augtype=FLAGS.augtype, n_samples=FLAGS.n_samples_aug,  
                            mp=mp, trans_scale=FLAGS.trans_scale, inp_channels=inp_channels,
                            n_mixtures=FLAGS.n_mixtures,loc_std=FLAGS.loc_std,
                            num_layers=FLAGS.num_flow_layers,
                            tanh_width= FLAGS.tanh_width, padding_mode=FLAGS.padding_mode).to(FLAGS.device)

    if FLAGS.batch_size <= 0:  # full batch
        batch_size = subset_size
    else:
        batch_size = min(FLAGS.batch_size, subset_size)

    train_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=batch_size, shuffle=True, detach=True)
    valid_loader = TensorDataLoader(X_test, y_test, transform=augmenter, batch_size=FLAGS.valid_batch_size, detach=True)

    # model
    if 'mnist' in FLAGS.dataset:
        optimizer = 'Adam'
        prior_structure = 'scalar'
        if FLAGS.model == 'mlp':
            model = MLP(28*28, width=1000, depth=1, output_size=10, fixup=False, activation='tanh')
        elif FLAGS.model == 'cnn':
            model = LeNet(in_channels=1, n_out=10, activation='tanh', n_pixels=28)
        else:
            raise ValueError('Unavailable model')
    elif 'cifar10' in FLAGS.dataset:
        optimizer = 'SGD'
        prior_structure = 'layerwise'  # for fixup params
        if FLAGS.model == 'cnn':
            model = LeNet(in_channels=3, activation='relu', n_pixels=32)
        elif FLAGS.model == 'resnet_8_8':
            model = ResNet(depth=8, num_classes=10, in_planes=8, in_channels=3)
        elif FLAGS.model == 'resnet_8_16':
            model = ResNet(depth=8, num_classes=10, in_planes=16, in_channels=3)
        elif FLAGS.model == 'resnet_14_16':
            model = ResNet(depth=14, num_classes=10, in_planes=16, in_channels=3)
        elif FLAGS.model == 'wrn':
            model = WideResNet()
        elif FLAGS.model == 'layer13s':
            model = layer13s()
        else:
            raise ValueError('Unavailable model')

    model.to(FLAGS.device)

    result = dict()
    if FLAGS.method == 'augerino':
        if hasattr(train_loader, 'attach'):
            train_loader.attach()
        model, losses, valid_perfs, aug_history = flow_invariance(
            model, train_loader, valid_loader, n_epochs=FLAGS.n_epochs, lr=FLAGS.lr,
            augmenter=augmenter, aug_reg=FLAGS.augerino_reg, lr_aug=FLAGS.lr_augerino,
            lr_min=FLAGS.lr_min, scheduler='cos', optimizer=optimizer, use_logp=FLAGS.use_logp, 
            burnin=FLAGS.n_epochs_burnin, wandb=wandb, decay_epochs=FLAGS.decay_epochs, 
            stepLR=FLAGS.stepLR, stepLR_factor=FLAGS.stepLR_factor, schedule_aug=FLAGS.schedule_aug,
            clip_grad=FLAGS.clip_grad, test_n_samples=FLAGS.test_n_samples,
        )
        aug_params = parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
        logging.info(f'aug params: {aug_params}.')
        result['losses'] = losses

    else:
        raise ValueError(f'Invalid method {FLAGS.method}')

    if FLAGS.save:
        result_path = Path(f'results/{FLAGS.dataset}/{FLAGS.model}/')
        result['flags'] = FLAGS.flag_values_dict()
        result['valid_perfs'] = valid_perfs
        file_name = f'flow_invariance_{FLAGS.method}_E={FLAGS.n_epochs}_N={FLAGS.subset_size}_S={FLAGS.n_samples_aug}_seed={FLAGS.seed}.pkl'

        result_path.mkdir(parents=True, exist_ok=True)
        with open(result_path / file_name, 'wb') as f:
            pickle.dump(result, f)
            
    if FLAGS.save_model is not None:
        os.makedirs(FLAGS.save_model, exist_ok=True)
        torch.save(model, FLAGS.save_model + '/model.pt')
        try:
            torch.save(augmenter, FLAGS.save_model + '/augmenter.pt')
        except e:
            logging.info(f'Failed to save augmenter: {e}')


if __name__ == '__main__':
    app.run(main)