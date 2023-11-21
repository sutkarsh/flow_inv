import torch
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
import traceback
import os
import coloredlogs,logging
from typing import Tuple, Callable
import enum

LOG = logging.getLogger('base')

def setup_logger(LOG, checkpoint_dir, debug=True):
    '''set up logger'''
    LOG.propagate = False
    coloredlogs.install(level='DEBUG', logger=LOG)
    formatter = logging.Formatter('%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    if debug:
        LOG.setLevel(logging.DEBUG)

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir,"run.log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    LOG.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

last_summary = [] # Tracks the last successful input to odeint; used only for debugging
def badvalcheck(f, label='NaN'):
    def checker(x, name, err=True, trace=False):
        if f(x):
            LOG.error(f"Bad value ({label}) detected in tensor {name}")
            if trace:
                traceback.print_exc()
            if err: 
                raise ValueError
    return checker

nancheck = badvalcheck(lambda x: x is not None and type(x) in [np.ndarray, torch.Tensor] and torch.isnan(x).any(), label='NaN')
infcheck = badvalcheck(lambda x: x is not None and type(x) in [np.ndarray, torch.Tensor] and torch.isinf(x).any(), label='Infinity')

def checkfunc(f, checks=[infcheck, nancheck], err=True, trace=True):
    '''
    A decorator to check for NaNs and Infs in the output of a function
    '''
    apply_checks = lambda x, name: [check(x, name, err=err, trace=trace) for check in checks]
    def wrapper(*args, **kwargs):
        for arg in args:
            apply_checks(arg, f"input to {f.__name__}")
        for key in kwargs:
            apply_checks(kwargs[key], f"input {key} to {f.__name__}")
        out = f(*args, **kwargs)
        if type(out) in [list, tuple]:
            for o in out:
                apply_checks(o, f"output of {f.__name__}")
        else:
            apply_checks(out, f"output of {f.__name__}")
        return out
    return wrapper

def checkmethod(f, checks=[infcheck, nancheck], err=True, trace=True):
    '''
    A decorator to check for NaNs and Infs in the input/output of a function
    '''
    apply_checks = lambda x, name: [check(x, name, err=err, trace=trace) for check in checks]
    def wrapper(self, *args, **kwargs):
        for i, arg in enumerate(args):
            apply_checks(arg, f"input {i} (arg: {arg}) to method {f.__name__} of object {self}")
        for key in kwargs:
            apply_checks(kwargs[key], f"input {key} to method {f.__name__} of object {self}")
        out = f(self, *args, **kwargs)
        if type(out) in [list, tuple]:
            for o in out:
                apply_checks(o, f"output of method {f.__name__} of object {self}")
        else:
            apply_checks(out, f"output of method {f.__name__} of object {self}")
        return out
    return wrapper



def expm(A,rtol=1e-4):
    """ Compute the matrix exponential of A
        assumes A has shape (bs,d,d)
        returns exp(A) with shape (bs,d,d) """
    I = torch.eye(A.shape[-1],device=A.device,dtype=A.dtype)[None].repeat(A.shape[0],1,1)
    norm = torch.norm(A, dim=(1,2))
    norm_clipped = torch.clamp(norm, max=15)
    A = A * norm_clipped[:, None, None]/(norm[:, None, None]+1e-5) # scale A by norm
    nancheck(A, "ODEint A")
    infcheck(A, "ODEint A")
    try:
        sol = torch.matrix_exp(A)
        global last_summary # Hacky way to maintain a summary of the last successful input to odeint; used only for debugging
        last_summary = [A.min().item(), A.mean().item(), A.max().item()]
        return sol
    except:
        LOG.error("odeint failed")
        LOG.info(f"Got input: min {A.min()}, mean {A.mean()} max {A.max()}")
        LOG.info(f"Last summary: min {last_summary[0]}, mean {last_summary[1]}, max {last_summary[2]}")
        raise Exception("odeint failed")

def get_rot_mat(theta):
    return torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta), theta*0]),
                        torch.stack([torch.sin(theta), torch.cos(theta), theta*0])], axis=0)

grid_sample = torch.nn.functional.grid_sample

def rot_img(x, theta):
    """
    Rotate image x by theta (degrees)
    :param x: image tensor of shape (bs, c, h, w)
    :param theta: rotation angle in degrees
    """
    if type(theta) in [float, int]:
        theta = torch.tensor(theta).to(x.device)
    theta = theta*np.pi/180.0
    rot_mat = get_rot_mat(theta)[None, ...].repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True)
    x = grid_sample(x, grid, align_corners=True)
    return x


def rot_img_batch(x, theta, padding_mode='zeros'):
    """
    Rotate image x by theta (degrees)
    :param x: image tensor of shape (bs, c, h, w)
    :param theta: rotation angle in degrees (bs,)
    """
    theta = (theta).to(x.device)
    theta = theta*np.pi/180.0
    rot_mat = get_rot_mat(theta).permute(2,0,1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True)
    x = grid_sample(x, grid, align_corners=True, padding_mode=padding_mode)
    return x


class PID:
    """
    A simple PID controller.
    Parameters:
        gain_factors: a tuple of three floats, [Kp, Ki, Kd]
        error_function: a function that takes the current value returns the error.
        integral_init: the initial value of the integral the I term.
    """
    def __init__(self, 
                gain_factors: Tuple[float, float, float] = (None, None, None),
                err_fn: Callable = None,
                output_range: Tuple[float, float] = (None, None),
                integral_init: float = 10.0
                ):
        self.kp, self.ki, self.kd = gain_factors
        self.err_fn = err_fn 
        self.output_range = output_range
        self.input_ema = EMA(0.9)
        self.output_ema = EMA(0.9)
        self.integral_init = integral_init
        self.reset()
    
    def reset(self):
        self.last_error = None
        self.integral = self.integral_init
        
    def step(self, value):
        value = self.input_ema.step(value)
        error = self.err_fn(value)
        self.integral += error
        derivative = 0 if self.last_error is None else error - self.last_error
        self.last_error = error
        output = self.kp*error + self.ki*self.integral + self.kd*derivative
        clipped_output = np.clip(output, self.output_range[0], self.output_range[1])
        clipped_output = self.output_ema.step(clipped_output) # smooth the output to avoid oscillations
        return clipped_output
        
class EMA:
    """
    A simple exponential moving average tracker.
    Parameters:
        alpha: the smoothing factor
    """
    def __init__(self, smoothing_factor: float = 0.9):
        self.smoothing_factor = smoothing_factor
        self.last_value = None
        
    def step(self, value):
        if self.last_value is None:
            self.last_value = value
        else:
            self.last_value = (1-self.smoothing_factor)*value + self.smoothing_factor*self.last_value
        return self.last_value
    
    @property
    def value(self):
        return self.last_value


class PIDerrortype(enum.Enum):
    """ The type of error to use in the PID controller """
    POINT = enum.auto()
    INTERVAL = enum.auto()
    INTERVAL_HUBER = enum.auto()
    
def target_error_fn(target_interval: Tuple[float, float], error_type='point'):
    """
    creates an error function for a target interval.
    parameters:
        target_interval: a tuple of two floats, the lower and upper bounds of the target interval.
        error_type (optional): 'point' or 'interval' or 'interval_huber' (we only use point)
    """
    midpoint = (target_interval[0]+target_interval[1])/2
    width = target_interval[1]-target_interval[0]
    radius = width/2
    def err_fn(value):
        difference = midpoint - value
        if error_type==PIDerrortype.POINT:
            # Aims for center of target interval
            return difference
        elif error_type==PIDerrortype.INTERVAL:
            # Aims for the target interval
            return np.maximum(0, np.abs(difference)-radius)*np.sign(difference)
        elif error_type==PIDerrortype.INTERVAL_HUBER:
            # Huber until width/2, then linear
            scaled_difference = np.abs(difference/radius)
            return np.where(scaled_difference<1, scaled_difference/2, 1-(0.5/scaled_difference))*difference
        else:
            raise ValueError(f"PID: Unknown 'error_type' {error_type}")    
    return err_fn
