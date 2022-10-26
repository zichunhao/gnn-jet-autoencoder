from argparse import Namespace
import argparse
from pathlib import Path
import torch
from utils.argparse_utils import (
    parse_data_settings, 
    parse_eval_settings, 
    parse_model_settings, 
    get_device, 
    get_dtype, 
    get_bool
)
from utils.initialize import initialize_models, initialize_test_dataloader
from utils.permutation import PermutationTest

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    logging.info(f"{args=}")

    # Loading data and initializing models
    test_loader = initialize_test_dataloader(
        paths=args.test_data_paths,
        batch_size=args.test_batch_size,
        vec_dims=args.vec_dims
    )

    encoder, decoder = initialize_models(args)
    logging.debug(f"{encoder=}")
    logging.debug(f"{decoder=}")
    logging.info(f'Latent space size: {encoder.latent_space_size}')
    logging.info(f'Compression rate: {encoder.latent_space_size / (args.vec_dims * args.num_jet_particles)}')
    
    logging.info("Running permutation test before training...")
    permutation_test = PermutationTest(
        encoder=encoder, 
        decoder=decoder,
        device=args.device,
        dtype=args.dtype
    )

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        logging.info('The models are initialized on GPU...')
    # Both on cpu
    else:
        logging.info('The models are initialized on CPU...')

    perm_result = permutation_test(test_loader, verbose=False)
    logging.info(f"Permutation invariance: {perm_result['invariance']}")
    logging.info(f"Permutation equivariance: {perm_result['equivariance']}")
    
def setup_argparse() -> Namespace:
    parser = argparse.ArgumentParser(description='GNN autoencoder training options')
    parser = parse_model_settings(parser)
    
    parser.add_argument('--test-data-paths', type=str, nargs='+', 
                        help='Paths of the test data.')
    parser.add_argument('-tbs', '--test-batch-size', type=int, default=128, metavar='',
                        help='Test batch size.')
    parser.add_argument('--unit', type=str, default='TeV',
                        help="The unit of momenta. Choices: ('GeV', 'TeV'). Default: TeV. ")
    parser.add_argument('--abs-coord', type=get_bool, default=True, metavar='',
                        help='Whether the data is in absolute coordinates. False when relative coordinates are used.')
    parser.add_argument('--polar-coord', type=get_bool, default=False, metavar='',
                        help='Whether the data is in polar coordinates (pt, eta, phi). False when Cartesian coordinates are used.')
    parser.add_argument('--normalized', type=get_bool, default=False, metavar='',
                        help='Whether the data is normalized. False when unnormalized data is used.')
    
    parser.add_argument('--device', type=get_device, default=get_device('-1'), metavar='',
                        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--dtype', type=get_dtype, default=torch.float64, metavar='',
                        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: torch.float64")

    parser.add_argument('--load-path', type=str, required=True, metavar='',
                        help='Path of the trained model to load.')
    parser.add_argument('--load-epoch', type=int, default=-1, metavar='',
                        help='Epoch number of the trained model to load.')

    args = parser.parse_args()
    logging.debug(f"args before updating: {args}")
    args.load_to_train = True

    return args
    
    

if __name__ == '__main__':
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)
