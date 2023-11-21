
import torch.nn as nn
import torch.nn.functional as F
from learned_inv.aug import *

count_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class SimpleConv(nn.Module):
    ''' Returns a 5-layer CNN with width parameter c. Same as Augerino, but no batchnorm'''
    def __init__(self, c=64, num_classes=10, in_channel=3, mp=4) -> None:
        super(SimpleConv, self).__init__()
        self.net =  nn.Sequential(
                                # Layer 0
                                nn.Conv2d(in_channel, c, kernel_size=3, stride=1,
                                        padding=1, bias=True),
                                # nn.BatchNorm2d(c),
                                nn.ReLU(),

                                # Layer 1
                                nn.Conv2d(c, c*2, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                                # nn.BatchNorm2d(c*2),
                                nn.ReLU(),
                                nn.MaxPool2d(2),

                                # Layer 2
                                nn.Conv2d(c*2, c*4, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                                # nn.BatchNorm2d(c*4),
                                nn.ReLU(),
                                nn.MaxPool2d(2),

                                # Layer 3
                                nn.Conv2d(c*4, c*8, kernel_size=3,
                                        stride=1, padding=1, bias=True),
                                # nn.BatchNorm2d(c*8),
                                nn.ReLU(),
                                nn.MaxPool2d(2),

                                # Layer 4
                                nn.MaxPool2d(mp),
                                Flatten(),
                                nn.Linear(c*8, num_classes, bias=True)
                            )

    def forward(self, x):
        if len(x.shape) == 5:
            # N x Aug x C x H x W
            N, A, C, H, W = x.shape
            out = self.net(x.view(N*A, C, H, W))
            return out.view(N, A, *out.shape[1:])
        else:
            return self.net(x)
    
    

class align_model(nn.Module):
    def __init__(self, aug,
                 in_channel=3, mp=4, n_copies=1,
                 pose_emb_netwidth=32, pose_emb_dimension=32, 
                 classifier_width=64, num_classes=4):
        """
        A basic alignment model for conditional pose learning.
        """
        super(align_model, self).__init__()
    
        
        self.classifier = SimpleConv(c=classifier_width, num_classes=num_classes, mp=mp, in_channel=in_channel).cuda()
        self.pose_embedder = SimpleConv(c=pose_emb_netwidth, num_classes=pose_emb_dimension, in_channel=in_channel, mp=mp).cuda()
        self.augmenter = aug
        self.n_copies = n_copies
        self.add_module('augmenter', self.augmenter)
        
        LOG.info(f"Classifier parameter count: {count_params(self.classifier)/1e6:.2f}M")
        LOG.info(f"Pose Embedder parameter count: {count_params(self.pose_embedder)/1e6:.2f}M")
        LOG.info(f"Augmenter parameter count: {count_params(aug)/1e6:.2f}M")
        
    def classify(self, x):
        return self.classifier(x)
    
    def embed(self, x):
        return self.pose_embedder(x)
    
    def augment(self, x, n_copies = 1, concat=False):
        emb = self.pose_embedder(x)
        
        all_augmented = []
        all_logp = []
        
        for i in range(n_copies):
            augmented, logp = self.augmenter(x, emb)
            all_augmented.append(augmented)
            all_logp.append(logp)
            
        if concat:
            out =  torch.stack(all_augmented, dim=1)
            return out
        return all_augmented, all_logp
    
    

class align_model_synch(nn.Module):
    def __init__(self, aug,
                 in_channel=3, mp=4, n_copies=1,
                 pose_emb_netwidth=32, pose_emb_dimension=32, 
                 classifier_width=64, num_classes=4, mlp=False):
        """
        A basic alignment model for conditional pose learning.
        Unlike align_model, this model uses synchronized augmentation by simply repeating the input
        """
        super(align_model_synch, self).__init__()

        if mlp:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3*32*32, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 4),
            )
        else:
            self.classifier = SimpleConv(c=classifier_width, num_classes=num_classes, mp=mp, in_channel=in_channel).cuda()
        self.pose_embedder = SimpleConv(c=pose_emb_netwidth, num_classes=pose_emb_dimension, in_channel=in_channel, mp=mp).cuda()
        self.augmenter = aug
        self.n_copies = n_copies
        self.add_module('augmenter', self.augmenter)
        
        LOG.info(f"Classifier parameter count: {count_params(self.classifier)/1e6:.2f}M")
        LOG.info(f"Pose Embedder parameter count: {count_params(self.pose_embedder)/1e6:.2f}M")
        LOG.info(f"Augmenter parameter count: {count_params(aug)/1e6:.2f}M")
        
    def classify(self, x):
        return self.classifier(x)
    
    def embed(self, x):
        return self.pose_embedder(x)
    
    def augment(self, x, n_copies = 1, concat=False):
        emb = self.pose_embedder(x)
        bs, c, h, w = x.shape
        
        # Now split the augmented and logp into n_copies
        if n_copies == 0:
            return [], []
        
        x = x.repeat(n_copies, 1, 1, 1)
        emb = emb.repeat(n_copies, 1,)
        augmented, logp = self.augmenter(x, emb)
        
        if n_copies == 1:
            all_augmented = [augmented]
            all_logp = [logp]
        else:
            all_augmented = augmented.split(bs, dim=0)
            all_logp = logp.split(bs, dim=0)
        
        if concat:
            out =  torch.stack(all_augmented, dim=1)
            return out
        return all_augmented, all_logp