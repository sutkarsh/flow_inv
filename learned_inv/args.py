import enum
from dataclasses import dataclass, field
from learned_inv.utils import *
from learned_inv.aug import AugType, BaseType

class dataset_choice(enum.Enum):
    MNIST = enum.auto()
    MARIO = enum.auto()
    MNIST_LT = enum.auto()
    SVHN = enum.auto()

class method_choice(enum.Enum):
    # Which method to use
    AUGERINO = enum.auto()
    INSTAAUG = enum.auto()
    NFAUG = enum.auto()

@dataclass(frozen=True)
class model_config:
    aug_type : AugType = AugType.ROTATE_TRANS # Augmentation type
    base_type : BaseType = BaseType.CONDITIONAL_GAUSSIAN_MIX
    trans_scale : float = 0.1
    n_mixtures: int = 1
    n_copies : int = 1
    aug_width : int = 32
    pose_emb_netwidth: int = 32
    pose_emb_dimension: int = 32
    classifier_width: int = 32
    nf_layers: int = 12
    nf_width: int = 32
    loc_std : float = 0.1
    gumbel_tau : float = 0.05
    hard_gs : bool = False
    tanh_width : float = 1.5
    logp_sq: bool = False
    ignore_intermediate_tanh : bool = False
    double_tap : bool = False

@dataclass(frozen=True)
class optimizer_config:
    bs : int = 128
    wd : float = 0
    epochs : int = 300
    lr : float = 1e-4

@dataclass(frozen=True)
class ent_controller_config:
    pid_warmup_epoch : int = 0
    ent_min : float = -1
    ent_max : float = -1
    k_p : float = 1e-2
    k_i : float = 1e-2
    k_d : float = 1e-2
    ema_decay : float = 0.9
    aug_loss_factor_min : float = 1e-4
    aug_loss_factor_max : float = 1
    err_fn_type : PIDerrortype = PIDerrortype.POINT

@dataclass(frozen=True)
class training_config:
    include_orig : bool = True
    no_aug_until : int = 0
    freeze_classifier : bool = False
    seed : int = 0
    dataset : dataset_choice = dataset_choice.MARIO
    data_rot: float = False
    mnist_classes : list = field(default_factory=lambda: [0,1,2,3,4,5,6,7,8,9])
    rot_range_factor : float = 2.0
    start_scale: float = 0.1
    n_modes : int = 1
    zero_mean_reg: float = 0
    weighted_resampling: bool = False
    class_size : int = None
    max_examples_per_class: int = 50000000
    finetune: bool = False
    freeze_augmenter: bool = False
    

@dataclass(frozen=True)
class Args:
    optimizer : optimizer_config = optimizer_config()
    trainer : training_config = training_config()
    ent_controller : ent_controller_config = ent_controller_config()
    model : model_config = model_config()
    ckpt_path : str = 'ckpt/'
    method: method_choice = method_choice.NFAUG
    load_model: str = None
    load_augmenter_only: bool = False
    no_train_augmenter: bool = False
    