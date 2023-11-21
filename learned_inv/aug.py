from learned_inv.flow import *
from enum import Enum

class AugType(Enum):
    """
    Enum for augmentation types
    """
    ROTATE = 0 # Only rotation
    ROTATE_TRANS = 1 # Rotation and translation
    FULL = 2 # Full affine; 6-dim
    TRANS = 3 # Only translation
    ROTGEN = 4 # 2-dimensional rotation generator; requires the model to produce skew symmetric coefficients   
    FULL_ROTGEN = 5 # Full affine; controls each element seperately
    CROP = 6 # Translation and scaling; only for insta-aug

class BaseType(Enum):
    """
    Enum for base distribution types (only gaussian and uniform supported for now)
    """
    UNIFORM = 0
    GAUSSIAN = 1
    UNIFORM_MIX = 2 # Mixture of uniform distributions
    GAUSSIAN_MIX = 3 # Mixture of gaussian distributions
    CONDITIONAL_GAUSSIAN_MIX = 4 # Mixture of gaussian distributions conditioned on input
    CONDITIONAL_UNIFORM_MIX = 5 # Mixture of uniform distributions conditioned on input
    CONDITIONAL_UNIFORM = 6 # Mixture of uniform distributions conditioned on input

class Aug(nn.Module):
    """
    Abstract class for augmenters.
    Subclass this to implement your own augmenter with a custom sample_weights method.
    """
    def __init__(self, trans_scale=0.1, aug_type=AugType.ROTATE):
        """
        Parameters:
            start_scale: float, initial scale of the flow (Not included in the abstract class)
            trans_scale: float, scale of the translation. Not sure how it is even used
        """
        super(Aug, self).__init__()
        self.trans_scale = trans_scale
        self.g0 = None
        
        nf_dims = {AugType.ROTATE: 1, AugType.TRANS: 2, AugType.ROTATE_TRANS: 3, AugType.FULL: 6, AugType.ROTGEN: 2, AugType.FULL_ROTGEN: 6, AugType.CROP: 4}[aug_type]
        self.nf_dims = nf_dims

    def sample_weights(self, c=None):
        """
        Sample weights.
        This is the heart of the Augmenter probability distribution.
        In principle, this can be any method that input x, context c, and returns weights and log_prob.
        
        Parameters:
            x: input tensor (Not included in the abstract class)
            c: context tensor
        Returns:
            weights: tensor, weights for the augmentations
            logp: tensor, log probability of the sample according to the model.
        """
        raise NotImplementedError


    def weights_to_affine(self, weights):
        generators = self.generate(weights)
        affine_matrices = expm(generators)
        return affine_matrices

    def apply_affine(self, x, affine_matrices):
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], 
                                 size = x.size(),
                                 align_corners=True)
        return F.grid_sample(x, flowgrid,align_corners=True, padding_mode=self.padding_mode)

    def format_weights(self, weights):
        # Take weights of shape (bs, nf_dims) and format them to be of shape (bs, 6), putting zeros where needed
        bs = weights.shape[0]
        if self.aug_type == AugType.ROTATE:
            weights = torch.cat([torch.zeros(bs, 2, device=weights.device), weights, torch.zeros(bs, 3, device=weights.device)], dim=-1)
        if self.aug_type == AugType.ROTATE_TRANS:
            weights = torch.cat([weights, torch.zeros(bs, 3, device=weights.device)], dim=-1)
        if self.aug_type == AugType.TRANS:
            weights = torch.cat([weights, torch.zeros(bs, 4, device=weights.device)], dim=-1)
        if self.aug_type == AugType.ROTGEN:
            weights = torch.cat([torch.zeros(bs, 2, device=weights.device), weights[...,:1], 
                                 torch.zeros(bs, 2, device=weights.device), weights[...,1:]], dim=-1)
            return torch.clamp(weights, min=-1, max=1) 
        weights = torch.clamp(weights, min=-1, max=1)
        weights = torch.cat([weights[:, :2], weights[:, 2:3]*np.pi, weights[:, 3:]], dim=1) # Scaling angles with pi
        return weights
    
    def transform(self, x, c = None):
        weights, logp = self.sample_weights(c)
        weights = self.format_weights(weights)
        affine_matrices = self.weights_to_affine(weights)
        x_out = self.apply_affine(x, affine_matrices)
        return x_out, logp

    def generate(self, weights):
        """
        return the sum of the scaled generator matrices
        """
        bs = weights.shape[0]

        if self.g0 is None or self.std_batch_size != bs:
            self.std_batch_size = bs

            ## tx
            self.g0 = torch.zeros(3, 3, device=weights.device)
            self.g0[0, 2] = 1. * self.trans_scale
            self.g0 = self.g0.unsqueeze(-1).expand(3,3, bs)

            ## ty
            self.g1 = torch.zeros(3, 3, device=weights.device)
            self.g1[1, 2] = 1. * self.trans_scale
            self.g1 = self.g1.unsqueeze(-1).expand(3,3, bs)

            self.g2 = torch.zeros(3, 3, device=weights.device)
            if self.aug_type not in [AugType.ROTGEN, AugType.FULL_ROTGEN]:
                self.g2[0, 1] = -1. # this creates a skew symmetric matrix whose expm is a rotation matrix
            self.g2[1, 0] = 1.
            self.g2 = self.g2.unsqueeze(-1).expand(3,3, bs)

            self.g3 = torch.zeros(3, 3, device=weights.device)
            if self.aug_type not in [AugType.ROTGEN, AugType.FULL_ROTGEN]:
                self.g3[0, 0] = 1. # This creates a diagonal matrix whose expm is a scaling matrix
            self.g3[1, 1] = 1.
            self.g3 = self.g3.unsqueeze(-1).expand(3,3, bs)

            self.g4 = torch.zeros(3, 3, device=weights.device)
            if self.aug_type not in [AugType.ROTGEN, AugType.FULL_ROTGEN]:
                self.g4[1, 1] = -1. # Shear
            self.g4[0, 0] = 1.
            self.g4 = self.g4.unsqueeze(-1).expand(3,3, bs)

            self.g5 = torch.zeros(3, 3, device=weights.device)
            if self.aug_type not in [AugType.ROTGEN, AugType.FULL_ROTGEN]:
                self.g5[1, 0] = 1. # Shear
            self.g5[0, 1] = 1.
            self.g5 = self.g5.unsqueeze(-1).expand(3,3, bs)

        out_mat  = weights[:, 0] * self.g0
        out_mat += weights[:, 1] * self.g1
        out_mat += weights[:, 2] * self.g2
        out_mat += weights[:, 3] * self.g3
        out_mat += weights[:, 4] * self.g4
        out_mat += weights[:, 5] * self.g5

        # transposes just to get everything right
        return out_mat.transpose(0, 2).transpose(2, 1)

    def forward(self, x, c = None):
        return self.transform(x, c)
    

class BaseDistributionAug(Aug):
    """Base distribution with a learned affine scale/translate layer. Not context dependent"""
    def __init__(self, start_scale=0.5, trans_scale=0.1,
                aug_type=AugType.ROTATE_TRANS,
                base_type=BaseType.UNIFORM,
                use_log_scale=True,
                n_mixtures=1,
                *args, **kwargs):

        super(BaseDistributionAug, self).__init__(trans_scale=trans_scale, aug_type=aug_type)

        self.start_scale = torch.tensor(start_scale)
        self.aug_type = aug_type
        self.softplus = nn.Softplus()
        self.use_log_scale = use_log_scale

        
        self.base_type = base_type

        if base_type == BaseType.UNIFORM:
            self.base = BaseUniform(self.nf_dims)
        elif base_type == BaseType.GAUSSIAN:
            self.base = BaseGaussian(self.nf_dims)
        elif base_type == BaseType.GAUSSIAN_MIX:
            self.base = GaussianMixture(n_mixtures, self.nf_dims)
        else:
            raise NotImplementedError

        flows = []

        # Construct flow model
        self.nf_model = nf.NormalizingFlow(self.base, flows)

        self.g0 = None
        self.std_batch_size = None

    def sample_weights(self, x):
        bs = x.shape[0]
        weights, logp = self.sample(bs)
        return weights, logp
    
    def sample(self, num_samples=1):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """
        z, log_q = self.nf_model.q0(num_samples)
        for flow in self.nf_model.flows:
            z, log_det = flow(z)
            log_q = log_q - log_det
        return z, log_q

class ConditionalBaseDistributionAug(Aug):
    """Base distribution with a learned affine scale/translate layer."""
    def __init__(self, trans_scale=0.1,
                pose_emb_dim = 32,
                aug_type=AugType.ROTATE_TRANS,
                base_type=BaseType.UNIFORM,
                n_mixtures=1,
                start_scale=0.1,
                *args, **kwargs):

        super(ConditionalBaseDistributionAug, self).__init__(trans_scale=trans_scale, aug_type=aug_type)

        self.aug_type = aug_type
        self.softplus = nn.Softplus()
        self.pose_emb_dim = pose_emb_dim # Width of the pose embedding used to condition the base distribution
        self.base_type = base_type

        if base_type == BaseType.CONDITIONAL_GAUSSIAN_MIX:
            self.base = ConditionalGaussianMixture(n_mixtures, self.nf_dims, scale=start_scale)
        else:
            raise NotImplementedError

        flows = []

        self.projection = nn.Linear(pose_emb_dim, self.base.param_len) # Project the pose embedding to the right size
        self.projection.weight.data *= 0.01
        self.projection.bias.data *= 0.01 # Initialize to small values
        
        # Construct flow model
        self.nf_model = ConditionalNormalizingFlow(self.base, flows) 

        self.g0 = None
        self.std_batch_size = None

    def sample_weights(self, input_embedding):
        bs = input_embedding.shape[0]
        base_q0_params_context = self.projection(input_embedding)
        weights, logp = self.nf_model.sample(input_embedding, base_q0_params_context)
        return weights, logp


class ConditionalFull(Aug):
    """Conditional full flow."""
    def __init__(self, start_scale=0.5, trans_scale=0.1,
                net: nn.Module = None,
                aug_type=AugType.ROTATE_TRANS,
                base_type=BaseType.UNIFORM,
                n_mixtures=1,
                *args, **kwargs):

        super(ConditionalBaseDistributionAug, self).__init__(trans_scale=trans_scale, aug_type=aug_type)

        self.start_scale = torch.tensor(start_scale)
        self.aug_type = aug_type
        self.net = net
        self.add_module("net", net)

        self.base_type = base_type

        if base_type == BaseType.CONDITIONAL_GAUSSIAN_MIX:
            self.base = ConditionalGaussianMixture(n_mixtures, self.nf_dims)
        else:
            raise NotImplementedError

        flows = []

        # Construct flow model
        self.nf_model = nf.NormalizingFlow(self.base, flows)

        self.g0 = None
        self.std_batch_size = None

    def sample_weights(self, x):
        context = self.net(x)
        weights, logp = self.sample(context)
        return weights, logp
    
    def sample(self, context):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """
        z, log_q = self.nf_model.q0(context)
        for flow in self.nf_model.flows:
            z, log_det = flow(z)
            log_q = log_q - log_det
        return z, log_q


class NFAug(Aug):
    """NFAug layer."""
    def __init__(self, trans_scale=0.1,
                pose_emb_dim = 32,
                aug_type=AugType.ROTATE_TRANS,
                base_type=BaseType.UNIFORM,
                n_mixtures=1,
                start_scale=0.1,
                num_layers=12,
                width=32,
                hard_gs=False,
                loc_std = 0.8,
                tanh_width = 2.0,
                ignore_intermediate_tanh = False,
                padding_mode='zeros',
                gumbel_tau = 0.5,
                *args, **kwargs
                ):
        super(NFAug, self).__init__(trans_scale=trans_scale, aug_type=aug_type)

        flow_scale_map = 'exp'
        self.padding_mode = padding_mode
        self.aug_type = aug_type
        self.pose_emb_dim = pose_emb_dim # Width of the pose embedding used to condition the base distribution        

        self.base_type = base_type

        if base_type == BaseType.CONDITIONAL_GAUSSIAN_MIX:
            self.base = ConditionalGaussianMixture(n_mixtures, self.nf_dims, scale=start_scale, hard_gumbel=hard_gs, loc_std=loc_std, tau=gumbel_tau)
        elif base_type == BaseType.UNIFORM:
            self.base = BaseUniform(self.nf_dims)
        elif base_type == BaseType.CONDITIONAL_UNIFORM:
            self.base = ConditionalUniform(self.nf_dims, scale=start_scale)
        elif base_type == BaseType.CONDITIONAL_UNIFORM_MIX:
            self.base = ConditionalUniformMixture(n_mixtures, self.nf_dims, scale=start_scale, loc_std=loc_std)
        else:
            raise NotImplementedError

        # Define list of flows
        self.num_layers = num_layers
        flows = []
        for i in range(num_layers):
            # use MLP
            big, small = int(np.ceil(self.nf_dims/2)), int(np.floor(self.nf_dims/2))
            mlp = nf.nets.MLP([big+pose_emb_dim, width, 2*small])
            mlp.net[-1].weight.data *= 0.01 # Initialize last layer to small values
            mlp.net[-1].bias.data *= 0.01
            param_map = ConditionalModel(mlp)
            # Add flow layer
            flows.append(ConditionalAffineCouplingBlock(param_map, scale_map=flow_scale_map))
            # Swap dimensions
            shuffle_mode = "shuffle" if self.nf_dims > 2 else "swap"
            flows.append(nf.flows.Permute(self.nf_dims, mode=shuffle_mode))
            if not ignore_intermediate_tanh:
                flows.append(TanhFlow(6))
        flows.append(TanhFlow(tanh_width))


        self.projection = nn.Linear(pose_emb_dim, self.base.param_len) # Project the pose embedding to the right size
        self.projection.weight.data *= 0.01
        self.projection.bias.data *= 0.01 # Initialize to small values
        
        # Construct flow model
        self.nf_model = ConditionalNormalizingFlow(self.base, flows) 

        self.g0 = None
        self.std_batch_size = None

    @checkmethod      
    def sample_weights(self, input_embedding):
        bs = input_embedding.shape[0]
        base_q0_params_context = self.projection(input_embedding)
        weights, logp = self.nf_model.sample(input_embedding, base_q0_params_context)
        logp_return = logp
        return weights, logp_return
    
    
class Augerino(Aug):
    """docstring for MLPAug"""
    def __init__(self, trans_scale=0.1, aug_type=AugType.ROTATE_TRANS,
                *args, **kwargs):
        super(Augerino, self).__init__(trans_scale=trans_scale, aug_type=aug_type)

        self.width = nn.Parameter(torch.zeros(self.nf_dims))
        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None
        self.aug_type = aug_type
        self.padding_mode = 'zeros'

    def set_width(self, vals):
        self.width.data = vals
        
    def sample_weights(self, x):
        bs = x.shape[0]
        weights = torch.rand(bs, self.nf_dims)
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width)
        weights = weights * width - width.div(2.)
        return weights, -(width).sum().repeat(bs)

class InstaAug(Aug):
    """NFAug layer."""
    def __init__(self, trans_scale=0.1, pose_emb_dim = 32, aug_type=AugType.ROTATE_TRANS, *args, **kwargs):
        super(InstaAug, self).__init__(trans_scale=trans_scale, aug_type=aug_type)

        self.pose_emb_dim = pose_emb_dim # Width of the pose embedding used to condition the base distribution

        self.projection = nn.Linear(pose_emb_dim, self.nf_dims*2) # Project the pose embedding to the right size
        self.projection.weight.data *= 0.01
        self.projection.bias.data *= 0.01 # Initialize to small values

        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None
        self.aug_type = aug_type
        self.padding_mode = 'zeros'

    @checkmethod      
    def sample_weights(self, input_embedding):
        proj = self.projection(input_embedding)
        widths, loc = proj[..., :self.nf_dims], proj[..., self.nf_dims:]
        widths = self.softplus(widths)
        logp = -torch.log(widths+1e-7).sum(dim=-1)
        
        z = (torch.rand_like(widths)*2 - 1)
        z = z.to(input_embedding.device, input_embedding.dtype)
        weights = z * widths + loc.clamp(-1, 1)
        
        return weights, logp
