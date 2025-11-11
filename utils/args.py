

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from utils.conf import base_path_dataset as base_path

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=0.03,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=42,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, help='Wandb project name')

    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--dataset_2', type=str, choices=['CIFAR10', 'CIFAR100', 'AUXImageNet100', 'TINYIMG'])
    parser.add_argument('--dataset_path', type=str, default=base_path())
    parser.add_argument('--forward_dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='The type of dataset used to define the heads for the next task.')
    parser.add_argument('--freezing_eval', choices=[None, 'training', 'buffer', 'training_and_buff'], default=None,
                    help='Validation set to be used during layers freezing.')
    parser.add_argument('--val_dataset_size', type=int, default=1000, 
                        help='The size of the validation set used for layers freezing evaluation.')
    parser.add_argument('--freezing_buff_size', type=int, default=200, 
        help='freezing buffer size')
    parser.add_argument('--dataset_subset', type=float, default=1., 
                        help='subset dataset to use.')
    


    # --- AUX batch-trimming: independent feature, off by default ---
    parser.add_argument('--aux_trim', type=int, default=0, choices=[0, 1],
        help='Enable independent AUX (dream) batch trimming (1=on). If off, training is unchanged.')
    parser.add_argument('--aux_keep_frac', type=float, default=0.5,
        help='Target fraction of AUX (dream) samples to keep in each batch when aux_trim=1. Range [0,1).')

    # --- SCDT / P‑SCDT (stealth‑constrained replay) ---
    parser.add_argument('--scdt', type=int, default=0, choices=[0, 1],
        help='Enable stealth‑constrained replay sampling (SCDT) for buffer replay.')
    parser.add_argument('--scdt_divergence', type=str, default='FAIRNESS',
        choices=['FAIRNESS', 'TV', 'KL'],
        help='Stealth ball: FAIRNESS (±δ per class), TV, or KL.')
    parser.add_argument('--scdt_budget', type=float, default=0.08,
        help='δ (FAIRNESS/TV) or ε (KL) bound on class-histogram drift.')
    parser.add_argument('--scdt_nominal', type=str, default='buffer',
        choices=['uniform', 'buffer', 'ema'],
        help='Nominal replay histogram π_nom (ER‑ACE typically uses uniform).')
    # Tiny preference tilt (optional; 0 = off). Keep this one knob for completeness.
    parser.add_argument('--scdt_pref_eta', type=float, default=0.5,
        help='Preference tilt strength η (0 = vanilla SCDT).')
    # (NEW) Within-class selection temperature for sample-level utilities
    parser.add_argument('--scdt_sample_eta', type=float, default=0.5,
        help='If >0, pick within-class samples probabilistically with weights ∝ exp(η * u_i); '
             'if 0, pick top-q by utility.')
    # (NEW) Momentum for class-loss EMA used to form utilities u (when available)
    parser.add_argument('--scdt_util_mom', type=float, default=0.9,
        help='EMA momentum for class-level utility (based on per-class CE).')
    # Optional temporal smoothing; default off keeps things simple/minimal.
    parser.add_argument('--scdt_window', type=int, default=0,
        help='Rolling window W for temporal stealth (0 disables).')
    
    parser.add_argument('--scdt_mass_mode', type=str, default='relative_to_aux',
    choices=['relative_to_aux', 'final_batch_frac'],
    help='How to compute AUX mass m. Default matches proposal: m=f*n_aux.')
    
    parser.add_argument('--scdt_ema_beta', type=float, default=0.1,
        help='EMA rate for nominal histogram when --scdt_nominal=ema.')



def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=32,
                        help='The batch size of the memory buffer.')
