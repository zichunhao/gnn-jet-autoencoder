from argparse import Namespace
import argparse
from pathlib import Path
import torch
import os.path as osp
from utils.utils import get_model_folder, get_best_epoch
from utils.argparse_utils import parse_data_settings, parse_eval_settings, parse_model_settings, parse_training_settings
from utils.initialize import initialize_dataloader, initialize_test_dataloader, initialize_models, initialize_optimizers
from utils.train import train_loop

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    logging.info(f"{args=}")

    # Loading data and initializing models
    train_loader, valid_loader = initialize_dataloader(
        paths=args.data_paths,
        batch_size=args.batch_size,
        vec_dims=args.vec_dims,
        train_fraction=args.train_fraction
    )

    encoder, decoder = initialize_models(args)
    logging.info(f"{encoder=}")
    logging.info(f"{decoder=}")
    logging.info(f'Latent space size: {encoder.latent_space_size}')
    
    if not args.load_to_train:
        import json
        outpath = get_model_folder(args)
        args_dir = outpath / "args_cache.json"
        with open(args_dir, "w") as f:
            json.dump({k: str(v) for k, v in vars(args).items()}, f)
    else:
        outpath = Path(args.load_path)
        # in case the folder has been deleted
        outpath.mkdir(parents=True, exist_ok=True)  
    logging.info(f"Output path: {outpath}")


    # trainings
    optimizer_encoder, optimizer_decoder = initialize_optimizers(args, encoder, decoder)

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        logging.info('The models are initialized on GPU...')
    # Both on cpu
    else:
        logging.info('The models are initialized on CPU...')

    logging.info(f'Training over {args.num_epochs} epochs...')

    '''Training'''
    train_loop(
        args, train_loader, valid_loader, encoder, decoder,
        optimizer_encoder, optimizer_decoder, outpath, args.device
    )

def setup_argparse() -> Namespace:
    parser = argparse.ArgumentParser(description='GNN autoencoder training options')
    parser = parse_data_settings(parser)
    parser = parse_training_settings(parser)
    parser = parse_eval_settings(parser)
    parser = parse_model_settings(parser)
    args = parser.parse_args()
    
    if args.load_to_train and args.load_epoch < 0:
        args.load_epoch = get_best_epoch(args.load_path, num=args.load_epoch)
    
    if args.patience <= 0:
        import math
        args.patience = math.inf
        
    return args
    
    

if __name__ == '__main__':
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)
