from functools import partial
from learned_inv.utils import *
import torch.nn as nn

import normflows as nf
from normflows.flows import Flow, ActNorm
from learned_inv.distributions import *


softplus = nn.Softplus()

class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    
    Based on the original version from normflows. My custom version allows for optional init scale and log scale
    """

    def __init__(self, shape, scale=True, shift=True, use_log_scale=True, init_scale=None):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param use_log_scale: Flag whether to use log scale
        :param init_scale: Float, initial scale value
        """
        super().__init__()
        self.use_log_scale = use_log_scale
        self.init_scale = init_scale
        
        param_generator = torch.zeros if self.use_log_scale else torch.ones
        
        if scale:
            self.s = nn.Parameter(param_generator(shape)[None])
        else:
            self.register_buffer("s", param_generator(shape)[None])

        if self.init_scale is not None:
            self.s.data.fill_(torch.log(self.init_scale) if self.use_log_scale else self.init_scale)

        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])

        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z):
        scale = torch.exp(self.s) if self.use_log_scale else softplus(self.s)
        z_ = z * scale + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_scale = self.s if self.use_log_scale else torch.log(scale+1e-8) # add small constant to avoid log(0)
        log_det = prod_batch_dims * torch.sum(log_scale)
        return z_, log_det

    def inverse(self, z):
        inv_scale = torch.exp(-self.s) if self.use_log_scale else 1/(softplus(self.s)+1e-8) # add small constant to avoid division by 0
        z_ = (z - self.t) * inv_scale
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_inv_scale = -self.s if self.use_log_scale else torch.log(inv_scale+1e-8) # add small constant to avoid log(0)
        log_det = prod_batch_dims * torch.sum(log_inv_scale)
        return z_, log_det

class TanhFlow(Flow):
    """   
    A flow layer which applies a tanh nonlinearity to the input. Useful for constraining the output to a certain range.
    """

    def __init__(self, alpha=1.0):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param use_log_scale: Flag whether to use log scale
        :param init_scale: Float, initial scale value
        """
        super().__init__()
        self.alpha = alpha
    def _log_d_tanh(self, z):
        return torch.sum(torch.log(1+ 1e-6 - torch.tanh(z)**2), dim=list(range(1, len(z.shape))))
    
    def forward(self, z):
        return torch.tanh(z/self.alpha)*self.alpha, self._log_d_tanh(z/self.alpha)

    def inverse(self, z):
        x = torch.arctanh(z/self.alpha)*self.alpha
        return x, -self._log_d_tanh(z/self.alpha)


class ConditionalModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.conditioning = True # Flag for conditional models; helps downstream models to know whether to pass conditioning variables
    def forward(self, x, y):
        '''
        Conditions on y by concatenating it to x
        '''
        z = torch.cat([x, y], dim=1)
        return self.base_model(z)


class ConditionalAffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    Same as in normflows, but allows for conditioning
    """

    def __init__(self, param_map, scale=True, scale_map="exp"):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid
        scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map
        self.conditioning = True # Flag for conditional models; helps downstream models to know whether to pass conditioning variables

    def forward(self, z, c=None):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        c is an optional conditioning variable that is passed to the param_map only if param_map has an attribute 'conditioning'
        """
        z1, z2 = z
        if hasattr(self.param_map, 'conditioning') and self.param_map.conditioning and c is not None:
            param = self.param_map(z1, c)
        else:
            param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

    def inverse(self, z, c=None):
        z1, z2 = z
        if hasattr(self.param_map, 'conditioning') and self.param_map.conditioning and c is not None:
            param = self.param_map(z1, c)
        else:
            param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 -= param
            log_det = 0
        return [z1, z2], log_det


class ConditionalAffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    Same as in normflows, but allows for conditioning
    """

    def __init__(self, param_map, scale=True, scale_map="exp", split_mode="channel"):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [nf.flows.Split(split_mode)]
        # Affine coupling layer
        self.flows += [ConditionalAffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [nf.flows.Merge(split_mode)]
        self.conditioning = True

    def forward(self, z, c = None):
        inp_norm = torch.norm(z)
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            if hasattr(flow, 'conditioning') and flow.conditioning and c is not None:
                z, log_det = flow(z, c)
            else:
                z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z, c = None):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            if hasattr(self.flows[i], 'conditioning') and self.flows[i].conditioning and c is not None:
                z, log_det = self.flows[i].inverse(z, c)
            else:
                z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


class ConditionalNormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        :param p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def sample(self, inp = None, q0_condition = None):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :param c: Conditioning variable (optional)
        :return: Samples, log probability
        """
        if hasattr(self.q0, 'conditioning') and self.q0.conditioning and inp is not None:
            z, log_q = self.q0(q0_condition)
        else:
            raise ValueError("q0 is not conditional!!")
            z, log_q = self.q0(inp)
        z, log_q = self.transform_noise(z, log_q, c=inp)
        return z, log_q
    

    def transform_noise(self, z, log_q, c = None):
        norms = []
        for i, flow in enumerate(self.flows):
            if hasattr(flow, 'conditioning') and flow.conditioning and c is not None:
                # print(f"conditioning {z.shape} {c.shape} {flow}")
                z, log_det = flow(z, c)
            else:
                z, log_det = flow(z)
            norms.append((i, z.norm().item()))
            log_q = log_q - log_det
        return z, log_q
    

    def log_prob(self, x, c = None, q0_condition = None):
        """
        Get log probability for batch
        :param x: Batch
        :param c: Conditioning variable (optional)
        :return: log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            if hasattr(self.flows[i], 'conditioning') and self.flows[i].conditioning and c is not None:
                z, log_det = self.flows[i].inverse(z, c)
            else:
                z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        if hasattr(self.q0, 'conditioning') and self.q0.conditioning and c is not None:
            log_q += self.q0.log_prob(z, q0_condition)
        else:
            log_q += self.q0.log_prob(z)
        return log_q