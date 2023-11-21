from learned_inv.utils import *
import torch.nn as nn
from enum import Enum

from normflows.distributions.base import BaseDistribution, GaussianMixture

class BaseGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution
    Acts as a base which can be augmented with scaling/translation with other layers
    """
    def __init__(self, shape, dtype=torch.float32, device="cuda"):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.temperature = None  # Temperature parameter for annealed sampling
        self.dtype = dtype
        self.device = device

    def forward(self, num_samples=1):
        z = torch.randn((num_samples,) + self.shape, dtype=self.dtype, device=self.device)
        log_p = self.log_prob(z)
        return z, log_p

    def log_prob(self, z):
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(0.5 * torch.pow(z, 2), list(range(1, self.n_dim + 1)))
        return log_p
    
    def prob(self, z):
        return torch.exp(self.log_prob(z))

class BaseUniform(BaseDistribution):
    """
    Distribution of a 1D random variable with uniform
    """

    def __init__(self, ndim, dtype=torch.float32, device="cuda"):
        """
        Constructor
        :param ndim: Int, number of dimensions
        :param scale: Iterable, width of uniform distribution
        """
        super().__init__()
        self.ndim = ndim
        # Set up indices and permutations
        self.ndim = ndim
        self.dtype = dtype
        self.device = device

    def forward(self, num_samples=1):
        z = self.sample(num_samples)
        return z, self.log_prob(z)

    def sample(self, num_samples=1):
        z = (
            torch.rand(
                (num_samples,self.ndim),
                device=self.device,
                dtype=self.dtype,
            )
            - 0.5 # shift to [-0.5, 0.5] because overall gap should be 1 in order for entropy to scale properly.
        )
        return z

    def log_prob(self, z):
        return torch.zeros_like(z)[:,:1]
    
    def prob(self, z):
        return torch.exp(self.log_prob(z)) * (z.abs() <= 0.5).float()

class ConditionalGaussianMixture(BaseDistribution):
    """
    Mixture of gaussians with diagonal covariance matrix
    """

    def __init__(self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True, loc_std=0.7, tau=0.5, hard_gumbel=False):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Uniform
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        :param loc_std: Float, standard deviation of the initial location parameters
        :param tau: temperature for gumbel softmax
        :param hard_gumbel: should we use hard gumbel softmax?
        """
        
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim
        self.tau = tau
        self.hard_gumbel = hard_gumbel

        if loc is None:
            loc = (np.random.rand(self.n_modes, self.dim)-0.5)*2*loc_std
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))*0.3
        elif np.isscalar(scale):
            scale = np.ones((self.n_modes, self.dim))*scale
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        self.conditioning = True

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc).float())
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)).float())
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)).float())
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc).float())
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)).float())
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)).float())
            
        self.param_len = self.n_modes * self.dim + self.n_modes * self.dim + self.n_modes
            
    def _context_to_params(self, context):
        bs = context.shape[0]
        assert context.shape[1] == self.param_len, f"Context shape ({context.shape}) does not match param len ({self.param_len})"
        loc, log_scale, weights = torch.split(context, [self.n_modes * self.dim, self.n_modes * self.dim, self.n_modes], dim=1)        
        loc = loc.reshape(bs, self.n_modes, self.dim)
        log_scale = log_scale.reshape(bs, self.n_modes, self.dim)
        weights = weights.reshape(bs, self.n_modes)
        
        return bs, loc + self.loc.clamp(-1, 1), log_scale + self.log_scale, weights + self.weight_scores

    def _z_to_logp(self, z, loc, log_scale, logw, detach_z_from_logp=False):
        # Compute log probability
        z_ = z.clone().detach() if detach_z_from_logp else z
        
        eps = (z_[:, None, :] - loc)/(torch.exp(log_scale)+1e-3) # Detach and add small value to prevent nans
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + logw
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)  
        return log_p  
        
    def forward(self, context, detach_z_from_logp=False):
        assert detach_z_from_logp in [True, False]
        # Get weights
        bs, loc, log_scale, weights = self._context_to_params(context)
        
        loc = torch.clamp(loc, -1, 1)
        if not self.hard_gumbel:
            log_scale = torch.clamp(log_scale, -2, torch.log(torch.tensor(2.0))) 
            weights = torch.clamp(weights, -1,1) # Clamping for stability
        
        logw = torch.log_softmax(weights, 1)
        
        mode_1h = torch.nn.functional.gumbel_softmax(logw, tau=self.tau, hard=self.hard_gumbel, dim=1)[..., None] # Should be (batch_size, n_modes, 1)
        
        # Get samples
        eps_ = torch.randn(
            bs, self.dim, dtype=loc.dtype, device=loc.device
        )
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1) 
        loc_sample = torch.sum(loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        log_p = self._z_to_logp(z, loc, log_scale, logw, detach_z_from_logp=detach_z_from_logp)

        return z, log_p

    def log_prob(self, z, context, pd=False):
        bs, loc, log_scale, weights = self._context_to_params(context)
        
        loc = torch.clamp(loc, -1, 1)
        if not self.hard_gumbel:
            log_scale = torch.clamp(log_scale, -2, torch.log(torch.tensor(2.0))) 
            weights = torch.clamp(weights, -1,1) # Clamping for stability
        
        logw = torch.log_softmax(weights.detach(), 1)
        
        log_p = self._z_to_logp(z, loc, log_scale, logw, detach_z_from_logp=False)
        
        return log_p

class ConditionalUniformMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True, new_mixture=False, loc_std=0.01
    ):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Gaussian
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        :param new_mixture: Flag, use the newer version of the mixture model
        :param loc_std: Float, standard deviation of the initial location parameters
        """
        
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim
        self.new_mix = new_mixture
        self.loc_std = loc_std

        if loc is None:
            loc = (np.random.rand(self.n_modes, self.dim)-0.5)*loc_std
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        self.conditioning = True
        
        # print(f"init --- loc: {loc.shape}, scale: {scale.shape}, weights: {weights.shape}")

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc).float())
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)).float())
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)).float())
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc).float())
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)).float())
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)).float())
            
        self.param_len = self.n_modes * self.dim + self.n_modes * self.dim + self.n_modes
            
    def _context_to_params(self, context):
        bs = context.shape[0]
        assert context.shape[1] == self.param_len, f"Context shape ({context.shape}) does not match param len ({self.param_len})"
        loc, log_scale, weights = torch.split(context, [self.n_modes * self.dim, self.n_modes * self.dim, self.n_modes], dim=1)        
        loc = loc.reshape(bs, self.n_modes, self.dim)
        log_scale = log_scale.reshape(bs, self.n_modes, self.dim)
        weights = weights.reshape(bs, self.n_modes)
        
        return bs, loc, log_scale, weights

    def forward(self, context):
        # Get weights
        bs, loc, log_scale, weights = self._context_to_params(context)
        
        loc = loc + self.loc.clamp(-1, 1)
        log_scale = log_scale + self.log_scale
        weights = weights + self.weight_scores
        
        log_scale = torch.clamp(log_scale, -6, torch.log(torch.tensor(2.0)))
        loc = torch.clamp(loc, -1, 1) 
        
        mode_1h = torch.nn.functional.gumbel_softmax(weights, tau=0.1, hard=True)[..., None]
        
        weights = torch.softmax(weights, 1)
        

        # Get samples
        eps_ = torch.rand(
            bs, self.dim, dtype=loc.dtype, device=loc.device
        ) - 0.5
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        
        z = eps_ * scale_sample + loc_sample
        
        # Compute log probability
        selected_modes_mask = ((z[:, None, :] - loc).abs()*2 <= torch.exp(log_scale)).prod(-1)
        log_p = (
            torch.log(torch.sum(selected_modes_mask * weights * torch.exp(-torch.sum(log_scale, 2)), -1))
        )

        return z, log_p

    def log_prob(self, z, context):
        # Get weights
        bs, loc, log_scale, weights = self._context_to_params(context)
        
        loc = loc + self.loc.clamp(-1, 1)
        log_scale = log_scale + self.log_scale
        weights = weights + self.weight_scores
        
        log_scale = torch.clamp(log_scale, -6, torch.log(torch.tensor(2.0)))
        loc = torch.clamp(loc, -1, 1) 
        weights = torch.softmax(weights, 1)
        
        # Compute log probability
        selected_modes_mask = ((z[:, None, :] - loc).abs()*2 <= torch.exp(log_scale)).prod(-1)
        log_p = (
            torch.log(torch.sum(selected_modes_mask * weights * torch.exp(-torch.sum(log_scale, 2)), -1))
        )

        return log_p




class ConditionalUniform(BaseDistribution):
    """
    Conditional Uniform Distribution
    """

    def __init__(self, dim, loc=None, scale=None, trainable=True):
        """
        Constructor
        :param dim: Number of dimensions
        :param loc: List of mean values
        :param scale: List of widths
        :param trainable: Flag, if true parameters will be optimized during training
        """
        
        # ? Step 1: can we broadcast the pararms to the batch size?
        # ? Step 2: Add context to the params
        
        super().__init__()

        self.dim = dim

        if loc is None:
            loc = (np.random.rand(self.dim)-0.5)
        loc = np.array(loc) 
        if scale is None:
            scale = np.ones((self.dim))
        scale = np.array(scale)
        self.conditioning = True
        
        # print(f"init --- loc: {loc.shape}, scale: {scale.shape}, weights: {weights.shape}")

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc).float())
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)).float())
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc).float())
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)).float())
            
        self.param_len = 2 * self.dim
            
    def _context_to_params(self, context):
        bs = context.shape[0]
        assert context.shape[1] == self.param_len, f"Context shape ({context.shape}) does not match param len ({self.param_len})"
        loc, log_scale = torch.split(context, [self.dim, self.dim], dim=1)        
        loc = loc.reshape(bs, self.dim)
        log_scale = log_scale.reshape(bs, self.dim)
        
        return bs, loc, log_scale

    def forward(self, context):
        # Get weights
        bs, loc, log_scale = self._context_to_params(context)
        
        loc = loc + self.loc.clamp(-1, 1)
        log_scale = log_scale + self.log_scale
        
        log_scale = torch.clamp(log_scale, -6, torch.log(torch.tensor(2.0))) # min range is close to float min, max range is 1 (width = 2)
        
        # Get samples
        eps_ = torch.rand(
            bs, self.dim, dtype=loc.dtype, device=loc.device
        ) - 0.5
        z = eps_ * torch.exp(log_scale) + loc

        log_p = (- torch.sum(log_scale, -1))
        return z, log_p

    def log_prob(self, z, context):
        # Get weights
        bs, loc, log_scale, = self._context_to_params(context)
        
        loc = loc + self.loc.clamp(-1, 1)
        log_scale = log_scale + self.log_scale
        
        
        log_scale = torch.clamp(log_scale, -6, torch.log(torch.tensor(2.0))) # min range is close to float min, max range is 1 (width = 2)
        
        log_p = (- torch.sum(log_scale, -1))

        return log_p


class ConditionalUniformMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True, new_mixture=False, loc_std=0.01
    ):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Gaussian
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        :param new_mixture: Flag, use the newer version of the mixture model
        :param loc_std: Float, standard deviation of the initial location parameters
        """
        
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim
        self.new_mix = new_mixture
        self.loc_std = loc_std

        if loc is None:
            loc = (np.random.rand(self.n_modes, self.dim)-0.5)*loc_std
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        self.conditioning = True
        
        # print(f"init --- loc: {loc.shape}, scale: {scale.shape}, weights: {weights.shape}")

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc).float())
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)).float())
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)).float())
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc).float())
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)).float())
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)).float())
            
        self.param_len = self.n_modes * self.dim + self.n_modes * self.dim + self.n_modes
            
    def _context_to_params(self, context):
        bs = context.shape[0]
        assert context.shape[1] == self.param_len, f"Context shape ({context.shape}) does not match param len ({self.param_len})"
        loc, log_scale, weights = torch.split(context, [self.n_modes * self.dim, self.n_modes * self.dim, self.n_modes], dim=1)        
        loc = loc.reshape(bs, self.n_modes, self.dim)
        log_scale = log_scale.reshape(bs, self.n_modes, self.dim)
        weights = weights.reshape(bs, self.n_modes)
        
        return bs, loc, log_scale, weights

    def forward(self, context):
        # Get weights
        bs, loc, log_scale, weights = self._context_to_params(context)
        
        loc = loc + self.loc.clamp(-1, 1)
        log_scale = log_scale + self.log_scale
        weights = weights + self.weight_scores
        
        log_scale = torch.clamp(log_scale, -6, torch.log(torch.tensor(2.0)))
        loc = torch.clamp(loc, -1, 1) 
        weights = torch.softmax(weights, 1)
        
        mode = torch.multinomial(weights, 1, replacement=True)[:, 0]
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None] # Should be (batch_size, n_modes, 1)

        # Get samples
        eps_ = torch.rand(
            bs, self.dim, dtype=loc.dtype, device=loc.device
        ) - 0.5
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        
        z = eps_ * scale_sample + loc_sample   
             
        log_p = (
        self.dim 
        + torch.log(weights)
        - torch.sum(log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z, context):
        # Get weights
        bs, loc, log_scale, weights = self._context_to_params(context)
        
        loc = loc + self.loc.clamp(-1, 1)
        log_scale = log_scale + self.log_scale
        weights = weights + self.weight_scores
        
        log_scale = torch.clamp(log_scale, -6, torch.log(torch.tensor(2.0)))
        loc = torch.clamp(loc, -1, 1) 
        weights = torch.softmax(weights, 1)
        
        # Sample mode indices
        mode = torch.multinomial(weights, 1, replacement=True)[:, 0]
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None] # Should be (batch_size, n_modes, 1)
        
        # Compute log probability
        selected_modes_mask = ((z[:, None, :] - loc).abs()*2 <= torch.exp(log_scale)).prod(-1)
        log_p = (
            # self.dim 
            # - torch.sum(torch.log(scale_sample), -1)
            torch.log(torch.sum(selected_modes_mask * weights * torch.exp(-torch.sum(log_scale, 2)), -1))
        )

        return log_p


base_distributions = {
    'uniform': BaseUniform,
    'gaussian': BaseGaussian,
    'gaussian_mixture': GaussianMixture,
    'conditional_gaussian_mixture': ConditionalGaussianMixture,
    'conditional_uniform_mixture': ConditionalUniformMixture,
    'conditional_uniform': ConditionalUniform,
}

