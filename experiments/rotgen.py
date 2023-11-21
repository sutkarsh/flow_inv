import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from data.generate_data import *
import tqdm
from learned_inv.model import *
from learned_inv.utils import *
from learned_inv.args import Args, dataset_choice, method_choice

import random
import tyro
from functools import partial, reduce
import pickle
import os

softplus = torch.nn.Softplus()
xent = torch.nn.CrossEntropyLoss(reduction='none')
logsoftmax = lambda x: x #nn.LogSoftmax(dim=-1)
rots = range(-180, 180, 5)

import torch.nn.functional as F

logs_dict = {} # Acculumates all the logged/plotted data. Written to disk every epoch.

def make_dist_plot(ax, model, testloader):
    return
    llim = []
    ulim = []
    med = []
    
    imp = model.pose_embedder
    aug = model.augmenter

    orig_inputs  = next(iter(testloader))[0].cuda().float()

    for ang in tqdm.tqdm(rots):
        inputs = rot_img(orig_inputs, ang)
        emb = imp(inputs).repeat(1000, 1)
        bs = emb.shape[0]
        weights, logp = aug.sample_weights(emb,)
        generators = aug.generate(weights)
        affine_matrices = expm(generators).to(weights.device)

        cos, sin = affine_matrices[:, 0, 0], affine_matrices[:, 1, 0]
        cos, sin = cos/(cos**2 + sin**2).sqrt(), sin/(cos**2 + sin**2).sqrt()
        angles = torch.arctan2(sin, cos) * 180 / np.pi
        
        angles= angles.reshape(1000, inputs.shape[0])

        llim.append(np.percentile(angles.detach().cpu().numpy(),1, axis=0))
        med.append(np.percentile(angles.detach().cpu().numpy(), 50, axis=0))
        ulim.append(np.percentile(angles.detach().cpu().numpy(), 99, axis=0))

    ax.plot(rots, np.array(llim)[..., 0], c=[0.7,0,0])
    ax.plot(rots, np.array(ulim)[..., 0], c=[0.7,0,0])
    ax.fill_between(rots, np.array(llim)[..., 0], np.array(ulim)[..., 0], color=[0.7,0,0], alpha=0.3, label="Limit Invariance")
    ax.plot(rots, np.array(med)[..., 0], c='black')
    major_ticks = np.arange(-180, 180, 90)
    minor_ticks = np.arange(-180, 180, 30)
    x_ticks = np.arange(-180, 180, 45)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xticks(x_ticks)
    ax.grid(which='both')
    ax.set_ylim(-180, 180)
    
    if "dist_plot" not in logs_dict:
        logs_dict["dist_plot"] = {}
        logs_dict["dist_plot"]["llim"] = []
        logs_dict["dist_plot"]["ulim"] = []
        logs_dict["dist_plot"]["med"] = []
    logs_dict["dist_plot"]["llim"].append(llim)
    logs_dict["dist_plot"]["ulim"].append(ulim)
    logs_dict["dist_plot"]["med"].append(med)
    

def track_dist_by_class(model, testloader, args):
    return
    imp = model.pose_embedder
    aug = model.augmenter

    testiter  = iter(testloader)
    inps, labs = [], []
    
    for _ in range(1):
        i, l = next(testiter)
        inps.append(i)
        labs.append(l)
    inps = torch.cat(inps, dim=0).cuda().float()
    labs = torch.cat(labs, dim=0).cuda().float()

    inputs = inps
    with torch.no_grad():
        orig_bs = inputs.shape[0]
        emb = imp(inputs).repeat(1000, 1)
        weights, logp = aug.sample_weights(emb)

    angles = weights[:, 2] * 180 / np.pi
    angles = angles.reshape(1000, orig_bs).transpose(0,1)
    pc1 = np.percentile(angles.detach().cpu().numpy(), 1, axis=-1, )
    pc99 = np.percentile(angles.detach().cpu().numpy(), 99, axis=-1, )
    width = (pc99-pc1)/2/180
        
    dist_width = {}
    for c in args.trainer.mnist_classes:
        idx = np.where(labs.detach().cpu().numpy() == c)[0]
        dist_width[c] = np.median(width[idx])

    if "width_by_class" not in logs_dict:
        logs_dict["width_by_class"] = []
    logs_dict["width_by_class"].append(dist_width)
    
    
def make_classwise_polar_plot(model, orig_inputs, lab, args, pth):
    return
    imp = model.pose_embedder
    aug =  model.augmenter
    
    orig_inputs = orig_inputs.to('cuda')
    
    classes = np.unique(lab)
    
    fig = plt.figure(figsize=(20,10))
    gs = fig.add_gridspec(2, len(classes))
    
    for i, c in enumerate(classes):        
        
        # Find random example of this class
        idx = np.random.choice(np.where(lab==c)[0])
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(orig_inputs[idx].permute(1,2,0).detach().cpu().numpy())
        ax.axis('off')

        # new subplot
        ax = fig.add_subplot(gs[1, i], projection='polar')
        print(f"Class {c}: {idx}, {lab[idx]}, orig_inputs.shape: {orig_inputs.shape}")
        make_polar_plot(ax, model, orig_inputs[idx:idx+1], args, ang=[0], tag=f"classwise_polar_class_{c}")
    plt.savefig(f"{pth}/classwise_polar.png")


def make_polar_plot(ax, model, orig_inputs, args : Args, ang=None, tag="polar_plot"):
    return
    imp = model.pose_embedder
    aug =  model.augmenter
    
    if ang is None:
        ang = [np.random.randint(-180/args.trainer.rot_range_factor , 180/args.trainer.rot_range_factor) for _ in range(5)]
    colors = ['purple', 'blue', 'black', 'pink', 'red']
    plotted = {}
    for a, c in zip(ang, colors):
        inputs = rot_img(orig_inputs[:1], a)
        emb = imp(inputs).repeat(10000, 1)
        bs = emb.shape[0]
        weights, logp = aug.sample_weights(emb)
        generators = aug.generate(weights)

        angles = weights[:, 2] * 180 / np.pi

        hist = np.histogram(angles.detach().cpu().numpy(), bins=rots, density=True)[0]
        plotted[a] = {'hist': hist, 'rots': rots}

        ax.plot((np.array(np.array(rots[:-1])+a)*np.pi/180),0.9*hist/hist.max(), label="Inv for one example", c=c)
    for a, c in zip(ang, colors):
        ax.scatter([a*np.pi/180], [0.97], c=c, s=40)
    ax.set_rmax(1)
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_rticks([])
    ax.set_theta_zero_location('N')
    
    if tag not in logs_dict:
        logs_dict[tag] = []
    logs_dict[tag].append(plotted)

def make_rot_plot(ax, model, orig_inputs, lab=None):
    ang = 0
    
    imp = model.pose_embedder
    aug =  model.augmenter
    classifier = model.classifier
    
    inputs = rot_img(orig_inputs[:1], ang)
    emb = imp(inputs).repeat(50000, 1)
    bs = emb.shape[0]
    weights, logp = aug.sample_weights(emb)
    weights = aug.format_weights(weights)


    generators = aug.generate(weights)
    angles = weights[:, 2] * 180 / np.pi
    probs = torch.exp(logp).detach().cpu().numpy()

    rots = np.linspace(-180,180,91)
    hist, bins = np.histogram(angles.detach().cpu().numpy(), bins=rots, density=True)
    ax.plot(rots[:-1], (hist/hist.sum())*25, label="Inv for one example", c='blue')
    
    loss_vals = []
    # Also adding a (normalized) loss plot in the same range
    for ang in np.linspace(-180, 180, 90):
        inputs = rot_img(torch.tensor(orig_inputs[:1]), torch.tensor(ang).float().cuda()).float().cuda()
        pred = logsoftmax(classifier(inputs))
        loss_vals.append(F.cross_entropy(pred, torch.tensor([lab]).to(pred.device)).mean().item())
    loss_vals = np.array(loss_vals)
    ax.plot(np.linspace(-180, 180, 90), loss_vals/loss_vals.max(), c='red', marker='o', label='loss')
    
    ax.grid()
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 180, 45))
    
    if "rot_plot" not in logs_dict:
        logs_dict["rot_plot"] = []
    logs_dict["rot_plot"].append({'hist': hist, 'rots': rots, 'loss': loss_vals})


def trainer(model, trainloader, testloader, args: Args):
    
    # sched = Scheduler(args.ent_min, args.ent_max, max_abs_status=args.max_abs_status, alpha=args.alpha, min_change_threshold=1e-3)
    pid_err_fn = target_error_fn((args.ent_controller.ent_min, args.ent_controller.ent_max), args.ent_controller.err_fn_type)
    pid_controller = PID(gain_factors=(args.ent_controller.k_p, args.ent_controller.k_i, args.ent_controller.k_d),
                err_fn=pid_err_fn,
                output_range=(args.ent_controller.aug_loss_factor_min, args.ent_controller.aug_loss_factor_max),   
            )
    ema = EMA(args.ent_controller.ema_decay)
    aug_loss_factor = 0
    
    # opt = lambda p: torch.optim.SGD(p, lr=args.classifier_lr, weight_decay=args.wd, momentum=0.9)
    optimizer =  torch.optim.AdamW(model.parameters(),  lr=args.optimizer.lr, weight_decay=args.optimizer.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.optimizer.epochs, eta_min=args.optimizer.lr/10)
    
    use_cuda = torch.cuda.is_available()
    logger = []

    plot_dict = {}
    
    model.classifier = torch.load('rotgen_mlp_classifier.pth')
    
    for epoch in tqdm.trange(args.optimizer.epochs, position=0, leave=False):  # loop over the dataset multiple times
        for i, data in tqdm.tqdm(enumerate(trainloader), position=1, leave=False, total=len(trainloader)):
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            if args.trainer.include_orig:
                original_xent_loss = xent(model.classifier(inputs), labels).mean()
            
            n_copies = args.model.n_copies if epoch >= args.trainer.no_aug_until else 0
            all_augmented, all_logp = model.augment(inputs, n_copies)
            all_preds = map(model.classifier, all_augmented)
            
            average_logp = reduce(lambda x, y: x + y, map(torch.mean, all_logp), torch.tensor(0.))/(n_copies+1e-8)
            
            all_aug_xent_losses = map(lambda pred: xent(pred, labels).mean(), all_preds)
            average_aug_xent_loss = reduce(lambda x, y: x + y, all_aug_xent_losses, torch.tensor(0.))/(n_copies+1e-8)

            # average_pred = reduce(sum, all_preds)/len(all_preds)
            # average_pred_xent = xent(average_pred, labels).mean()
    
            classification_loss = average_aug_xent_loss.mean()
            if args.trainer.include_orig:
                classification_loss = classification_loss + original_xent_loss.mean() 

            # print(f"Losses {classification_loss.item()} {average_logp.item()} {aug_loss_factor}")
            loss = classification_loss + aug_loss_factor * average_logp.mean()
            loss.backward()
            
            ent = -average_logp.mean()
            

            if epoch < args.trainer.no_aug_until:
                # Zero out gradients for aug and model
                model.augmenter.zero_grad(set_to_none=True)
                model.pose_embedder.zero_grad(set_to_none=True)
            elif args.trainer.freeze_classifier:
                model.classifier.zero_grad(set_to_none=True)

            optimizer.step()

            width_norm = ema.step(ent.item())

            if (epoch >= args.ent_controller.pid_warmup_epoch):
                aug_loss_factor  = pid_controller.step(width_norm)
            aug_loss_factor = np.clip(aug_loss_factor, args.ent_controller.aug_loss_factor_min, args.ent_controller.aug_loss_factor_max)
            # Logging

            log = [loss.item(), width_norm, aug_loss_factor]
            logger.append(log)
            
            update_dict = {'crossentloss': classification_loss.item(), 'totalloss': loss.item(), 'widthnorm': width_norm, 'auglossfactor': aug_loss_factor}
            if args.trainer.include_orig:
                update_dict['original_xent_loss'] = original_xent_loss.item()
            
            for k, v in update_dict.items():
                if k not in plot_dict:
                    plot_dict[k] = []
                plot_dict[k].append(v)

        # Test Accuracy
        test_acc = 0
        for i, data in enumerate(testloader):
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            preds = model.classifier(inputs)
            test_acc += (preds.argmax(dim=-1) == labels).sum().item()
        test_acc /= len(testloader.dataset)
        
        LOG.info(f"Epoch {epoch}: test_acc: {test_acc} crossentloss: {classification_loss.item()}, totalloss: {loss.item()}, widthnorm: {width_norm}, auglossfactor: {aug_loss_factor}")
        if epoch % 10 == 0:
            torch.save(model, f"{args.ckpt_path}/new_model.pt")
        plt.close('all')
        
        scheduler.step()

        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        gs = fig.add_gridspec(3, 4)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(plot_dict['crossentloss'], label='crossentloss', c='black')
        print("lenght", len(plot_dict['crossentloss']))
        if 'original_xent_loss' in plot_dict:
            ax1.semilogy(plot_dict['original_xent_loss'], label='original_xent_loss', c='red')
        ax1.grid()
        ax1.legend()
        
        ax1_5 = fig.add_subplot(gs[0, 1])
        ax1_5.semilogy(plot_dict['totalloss'], label='totalloss', c='black')
        print("lenght", len(plot_dict['totalloss']))
        ax1_5.grid()
        ax1_5.legend()
        
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(plot_dict['widthnorm'], label='ent')
        ax2.fill_between(range(len(plot_dict['widthnorm'])), np.ones_like(plot_dict['widthnorm'])*args.ent_controller.ent_min, np.ones_like(plot_dict['widthnorm'])*args.ent_controller.ent_max, alpha=0.2, color='purple')
        ax2.grid()
        ax2.legend()
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.semilogy(plot_dict['auglossfactor'], label='auglossfactor')
        ax3.grid()
        ax3.legend()
        
        if args.trainer.dataset == dataset_choice.MARIO:    
            imgs = np.load("./data/images.npz")
            inp = torch.FloatTensor(imgs['mario']).reshape(1, 3, 32, 32).cuda()    
            lab = torch.tensor(0).unsqueeze(0).cuda()
        else:
            k = np.random.randint(0, len(testloader.dataset))
            inp, lab = testloader.dataset[k]
            inp = inp.unsqueeze(0).cuda()
        # make_dist_plot(ax4, model, testloader)
        
        ax4 = fig.add_subplot(gs[1, 2:3])
        if args.method == method_choice.AUGERINO:
            ###
            weights, _ = model.augmenter.sample_weights(inp.repeat(1000, 1, 1, 1))
            w_ = weights.detach().cpu().numpy()
        elif args.method == method_choice.INSTAAUG:
            ###
            emb = model.pose_embedder(inp).repeat(1000, 1)
            weights, _ = model.augmenter.sample_weights(emb)
            w_ = weights.detach().cpu().numpy()
        else:
            emb = model.pose_embedder(inp).repeat(1000, 1)
            # weights = model.augmenter.format_weights(weights)
            base_q0_params_context = model.augmenter.projection(emb)
            weights, _ = model.augmenter.nf_model.sample(emb, base_q0_params_context)
            w_ = weights.detach().cpu().numpy()
        print(f"weights before format {weights.shape}")
        weights =  model.augmenter.format_weights(weights)
        print(f"weights after format {weights.shape}")
        affine_matrices = model.augmenter.weights_to_affine(weights)
        x_out = model.augmenter.apply_affine(inp.repeat(1000, 1, 1, 1), affine_matrices)
        cls = model.classifier(x_out)
        cls = torch.softmax(cls, dim=-1)
        print("cls", cls)
        cls = cls[:, lab].detach().cpu().numpy()
        print(cls.shape, cls.min(), cls.max())
        # ax4.scatter(w_[:, 0], w_[:, 1], c=cls, s=15)
        ax4.scatter(w_[:, 0], w_[:, 1], c=cls, s=15)
        ax4.set_title("Augmentation weights")
        ax4.set_xlim(-1, 1)
        ax4.set_ylim(-1, 1)
        ax4.legend()
        
        ax_angle = fig.add_subplot(gs[1, -1:])
        twobytwo = affine_matrices[:, :2, :2].detach().cpu()
        col1, col2 = twobytwo[:, :, 0], twobytwo[:, :, 1]
        cosine_similarity = (col1 * col2).sum(-1) / (col1.norm(dim=-1) * col2.norm(dim=-1))
        angle = torch.acos(cosine_similarity) * 180 / np.pi
        angle_deviation =((angle-90).abs()+1e-3)
        angle_deviation = angle_deviation.detach().cpu().numpy().flatten()
        hist, bin_edges = np.histogram(angle_deviation, bins=np.linspace(0, 90, 30), density=True)
        ax_angle.plot(bin_edges[:-1], hist, label='angle deviation', c='black', lw=3)
        ax_angle.grid()
        ax_angle.set_title("Angle deviation")
        
        
        ax5 = fig.add_subplot(gs[2, 0])
        emb = model.pose_embedder(inp)
        augmented, _ = model.augmenter(inp, emb)
        ax5.imshow(np.concatenate([inp[0].permute(1,2,0).detach().cpu().numpy(), augmented[0].permute(1,2,0).detach().cpu().numpy()], axis=1))
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[2, 1], projection='polar')
        make_polar_plot(ax6, model, inp, args)

        ax7 = fig.add_subplot(gs[2, 2:])
        make_rot_plot(ax7, model, inp, lab=lab)
        
        ax7.legend()

        plt.savefig(f"{args.ckpt_path}/plot.png")
        inp, lab = next(iter(testloader))
        make_classwise_polar_plot(model, inp, lab, args, args.ckpt_path)
        
        track_dist_by_class(model, testloader, args)
        
        logs_dict.update(plot_dict)
        pickle.dump(logs_dict, open(f"{args.ckpt_path}/logs_dict.pkl", "wb"))
        




def main(args : Args):
    setup_logger(LOG, "../logs/exp_01_unif/")
    LOG.info("args "+ str(args))

    # Random seed
    seed = args.trainer.seed
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

    dataset = args.trainer.dataset

    if dataset == dataset_choice.MNIST:
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())

        keep_classes = args.trainer.mnist_classes
        num_classes = len(keep_classes)
        
        # Filter train set
        keep_idx = []
        for i in range(len(trainset.targets)):
            if trainset.targets[i] in keep_classes:
                keep_idx.append(i)
        trainset = torch.utils.data.Subset(trainset, keep_idx)
        
        # Filter test set
        keep_idx = []
        for i in range(len(testset.targets)):
            if testset.targets[i] in keep_classes:
                keep_idx.append(i)
        testset = torch.utils.data.Subset(testset, keep_idx)
        
        LOG.info(f"Keeping classes {keep_classes} from MNIST dataset")
        LOG.info(f"Train set size {len(trainset)}, test set size {len(testset)}")
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.optimizer.bs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.optimizer.bs)
    elif dataset == dataset_choice.MARIO:
        ntrain = 12000
        ntest = 600
        trainloader, testloader = generate_mario_data(ntrain=ntrain-(ntrain % args.trainer.n_modes),
                                                      ntest=ntest-(ntest % args.trainer.n_modes),
                                                    batch_size=args.optimizer.bs, dpath="./data/",
                                                    rot_range=np.pi/args.trainer.rot_range_factor,
                                                    n_modes=args.trainer.n_modes)
        num_classes = 4
    else:
        raise ValueError("Invalid dataset")


    mp = 2 if dataset == dataset_choice.MNIST else 4
    in_channel = 1 if dataset == dataset_choice.MNIST else 3
    num_classes = 10 if dataset == dataset_choice.MNIST else 4
    pose_emb_dimension = args.model.pose_emb_dimension

    if args.method == method_choice.NFAUG:
        augmenter = NFAug(aug_type=args.model.aug_type,
                        base_type=args.model.base_type, 
                        n_mixtures=args.model.n_mixtures,
                        pose_emb_dim=pose_emb_dimension, 
                        start_scale = args.trainer.start_scale,
                        num_layers=args.model.nf_layers,
                        width=args.model.nf_width,
                        mlp_nonlin=None,
                        flow_scale_map='exp',
                        hard_gs=args.model.hard_gs,
                        loc_std=args.model.loc_std,
                        gumbel_tau=args.model.gumbel_tau,
                        tanh_width=args.model.tanh_width,
                        logp_sq=args.model.logp_sq,
                        ignore_intermediate_tanh=args.model.ignore_intermediate_tanh,
                        )
    elif args.method == method_choice.AUGERINO:
        augmenter = Augerino(aug_type=args.model.aug_type)
    elif args.method == method_choice.INSTAAUG:
        augmenter = InstaAug(aug_type=args.model.aug_type, pose_emb_dim=pose_emb_dimension)
    else:
        raise NotImplementedError("Invalid method")

    model = align_model_synch(in_channel = in_channel, mp = mp,
                        pose_emb_dimension=pose_emb_dimension, pose_emb_netwidth=32,
                        classifier_width=32, num_classes=num_classes,
                        aug = augmenter, mlp=True)
    
    model.cuda()

    os.makedirs(args.ckpt_path, exist_ok=True)

    trainer(model, trainloader, testloader, args)


args = tyro.cli(Args)
main(args)