from argparse import Namespace
from pathlib import Path
from typing import Iterable, List, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import logging
from utils.data import JetMomentaDataset
from models import Encoder, Decoder

def initialize_models(args):
    encoder = Encoder(
        num_nodes=args.num_jet_particles,
        input_node_size=args.vec_dims,
        latent_node_size=args.latent_node_size,
        node_sizes=args.encoder_node_sizes,
        edge_sizes=args.encoder_edge_sizes,
        num_mps=args.encoder_num_mps,
        alphas=args.encoder_alphas,
        batch_norm=args.encoder_batch_norm,
        latent_map=args.latent_map,
        dtype=args.dtype, 
        device=args.device
    )

    decoder = Decoder(
        num_nodes=args.num_jet_particles,
        latent_node_size=args.latent_node_size,
        output_node_size=args.vec_dims,
        node_sizes=args.decoder_node_sizes,
        edge_sizes=args.decoder_edge_sizes,
        num_mps=args.decoder_num_mps,
        alphas=args.decoder_alphas,
        latent_map=args.latent_map,
        normalize_output=args.normalized,
        batch_norm=args.decoder_batch_norm,
        dtype=args.dtype, 
        device=args.device
    )

    if args.load_to_train:
        model_path = Path(args.load_path)
        try:
            if (model_path / f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth').exists():
                encoder.load_state_dict(torch.load(
                    model_path / f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth',
                    map_location=args.device
                ))
                decoder.load_state_dict(torch.load(
                    model_path / f'weights_encoder/epoch_{args.load_epoch}_decoder_weights.pth',
                    map_location=args.device
                ))
            elif (model_path / f'weights_encoder/epoch_{args.load_epoch-1}_encoder_weights.pth').exists():
                # load the previous epoch's weights
                encoder.load_state_dict(torch.load(
                    model_path / f'weights_encoder/epoch_{args.load_epoch-1}_encoder_weights.pth',
                    map_location=args.device
                ))
                decoder.load_state_dict(torch.load(
                    model_path / f'weights_encoder/epoch_{args.load_epoch-1}_decoder_weights.pth',
                    map_location=args.device
                ))
            elif (model_path / f'weights_encoder/best.pth').exists():
                logging.warning(f"Epoch {args.load_epoch} Not found. Loading the best model instead of the specified epoch.")
                encoder.load_state_dict(torch.load(
                    model_path / 'weights_encoder/best_encoder_weights.pth',
                    map_location=args.device
                ))
                decoder.load_state_dict(torch.load(
                    model_path / 'weights_encoder/best_decoder_weights.pth',
                    map_location=args.device
                ))
            else:
                logging.warning(f"No model at epoch {args.load_epoch} found in {model_path}. Training from scratch.")
        except FileNotFoundError:
            logging.warning(f"No model at epoch {args.load_epoch} found in {model_path}. Training from scratch.")
    
    return encoder, decoder

def initialize_optimizers(
    args: Namespace, 
    encoder: Encoder, 
    decoder: Decoder
) -> Tuple[Optimizer, Optimizer]:
    if args.optimizer.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer_encoder = torch.optim.RMSprop(encoder.parameters(), lr=args.lr, eps=1e-16, momentum=0.9)
        optimizer_decoder = torch.optim.RMSprop(decoder.parameters(), lr=args.lr, eps=1e-16, momentum=0.9)
    elif args.optimizer.lower() == 'adagrad':
        optimizer_encoder = torch.optim.Adagrad(encoder.parameters(), lr=args.lr, eps=1e-16)
        optimizer_decoder = torch.optim.Adagrad(decoder.parameters(), lr=args.lr, eps=1e-16)
    elif args.optimizer.lower() == 'sgd':
        optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9)
        optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            "Other choices of optimizer are not implemented. "
            f"Available choices are 'Adam' and 'RMSprop'. Found: {args.optimizer}."
        )
    return optimizer_encoder, optimizer_decoder


def initialize_dataloader(
    paths: Union[List[Path], List[str], Path, str], 
    batch_size: int, 
    train_fraction: float = 0.65,
    vec_dims: int = 3,
) -> Tuple[DataLoader, DataLoader]:
    if isinstance(paths, Iterable):
        # load data from files
        data_train_list = []
        data_valid_list = []
        for path in paths:
            data = torch.load(path)
            # split data into train and validation
            split_idx = int(len(data) * train_fraction)
            data_train_list.append(data[:split_idx])
            data_valid_list.append(data[split_idx:])
        data_train = torch.cat(data_train_list, dim=0)
        data_valid = torch.cat(data_valid_list, dim=0)
    else:
        # load from a single file
        data = torch.load(paths)
        split_idx = int(len(data) * train_fraction)
        data_train = data[:split_idx]
        data_valid = data[split_idx:]
    
    # initialize datasets
    dataset_train = JetMomentaDataset(data_train, vec_dims=vec_dims)
    dataset_valid = JetMomentaDataset(data_valid, vec_dims=vec_dims)
    
    # initialize data loaders
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    
    logging.info(
        "Data for training loaded. "
        f"Number of training samples: {len(dataset_train)}. "
        f"Number of validation samples: {len(dataset_valid)}."
    )

    return loader_train, loader_valid


def initialize_test_dataloader(
    paths: Union[List[Path], List[str], Path, str], 
    batch_size: int, 
    vec_dims: int = 3
) -> DataLoader:
    if isinstance(paths, Iterable):
        # load data from files
        data_test_list = []
        for path in paths:
            data = torch.load(path)
            data_test_list.append(data)
        data = torch.cat(data_test_list, dim=0)
    else:
        data = torch.load(paths)
    jet_data = JetMomentaDataset(data, vec_dims=vec_dims)
    
    logging.info(
        "Data for testing loaded. "
        f"Number of testing samples: {len(jet_data)}."
    )
    
    return DataLoader(jet_data, batch_size=batch_size, shuffle=False)
