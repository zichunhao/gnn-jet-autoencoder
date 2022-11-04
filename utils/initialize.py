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
        dropout=args.encoder_dropout,
        batch_norm=args.encoder_batch_norm,
        latent_map=args.latent_map,
        dtype=args.dtype, 
        device=args.device
    )

    logging.info(f"Encoder initialized: {encoder}")

    decoder = Decoder(
        num_nodes=args.num_jet_particles,
        latent_node_size=args.latent_node_size,
        output_node_size=args.vec_dims,
        node_sizes=args.decoder_node_sizes,
        edge_sizes=args.decoder_edge_sizes,
        num_mps=args.decoder_num_mps,
        alphas=args.decoder_alphas,
        dropout=args.decoder_dropout,
        latent_map=args.latent_map,
        normalize_output=args.normalized,
        batch_norm=args.decoder_batch_norm,
        dtype=args.dtype, 
        device=args.device
    )
    
    logging.info(f"Decoder initialized: {decoder}")
    
    logging.info(f"Encoder parameters: {encoder.num_learnable_params}")
    logging.info(f"Decoder parameters: {decoder.num_learnable_params}")
    logging.info(f"Total parameters: {encoder.num_learnable_params + decoder.num_learnable_params}")

    if args.load_to_train:
        model_path = Path(args.load_path)
        logging.info(f"Loading model from {model_path} at epoch {args.load_epoch}.")
        try:
            if (model_path / f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth').exists():
                _load_weights(
                    encoder, 
                    path=model_path / f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth',
                    map_location=args.device
                )
                _load_weights(
                    decoder, 
                    model_path / f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth',
                    map_location=args.device
                )
                epoch = args.load_epoch
                logging.info(f"Loaded model from {model_path} at epoch {epoch}.")
           
            elif (model_path / f'weights_encoder/epoch_{args.load_epoch-1}_encoder_weights.pth').exists():
                logging.warning(f"No model at epoch {args.load_epoch} found in {model_path}. Searching for epoch {args.load_epoch - 1}.")
                
                # load the previous epoch's weights
                _load_weights(
                    encoder, 
                    model_path / f'weights_encoder/epoch_{args.load_epoch-1}_encoder_weights.pth',
                    map_location=args.device
                )
                _load_weights(
                    decoder,
                    model_path / f'weights_decoder/epoch_{args.load_epoch-1}_decoder_weights.pth',
                    map_location=args.device
                )
                epoch = args.load_epoch - 1
                logging.info(f"Loaded model from {model_path} at epoch {epoch}.")
            
            elif (model_path / f'weights_encoder/best_encoder_weights.pth').exists():
                logging.warning(f"No model at epoch {args.load_epoch - 1} found in {model_path}. Searching for best epoch.")
                logging.warning(f"Epoch {args.load_epoch} Not found. Loading the best model instead of the specified epoch.")
                _load_weights(
                    encoder,
                    model_path / 'weights_decoder/best_encoder_weights.pth',
                    map_location=args.device
                )
                _load_weights(
                    decoder,
                    model_path / 'weights_decoder/best_decoder_weights.pth',
                    map_location=args.device
                )
                logging.info(f"Loaded model from {model_path} at best epoch.")
            else:
                logging.warning(f"No model at best epoch found in {model_path} Training from scratch.")
                epoch = "None"
        
        except FileNotFoundError:
            logging.warning(f"No model at epoch {args.load_epoch} found in {model_path}. Training from scratch.")
    
    return encoder, decoder


def _load_weights(
    model: torch.nn.Module,
    path: Path,
    model_name: str = None,
    *args, **kwargs
) -> None:
    """Load weights from path."""
    model_name = model_name if model_name is not None else model.__class__.__name__
    try:
        model.load_state_dict(torch.load(path, *args, **kwargs))
    except RuntimeError as e:
        # new updates remove unused network in GraphNet
        logging.error(f'Error loading {model_name} weights from {path}: {e}.')
        logging.info('Loading weights with strict=False')
        model.load_state_dict(torch.load(path, *args, **kwargs), strict=False)
    logging.info(f'Weights {model_name} loaded from {path}.')

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
    logging.info(f"Loading from {paths}")
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
    logging.info(f"Loading from {paths}")
    if isinstance(paths, (list, tuple)):
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
