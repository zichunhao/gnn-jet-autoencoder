from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
import numpy as np
import math
import os.path as osp
import time
from models import Decoder, Encoder
from utils.utils import make_dir, save_data, plot_eval_results
from utils.const import EPS
from utils.jet_analysis import plot_p
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

BLOW_UP_THRESHOLD = 1e8

def train(
    args: Namespace, 
    loader: DataLoader, 
    encoder: Encoder, 
    decoder: Decoder, 
    optimizer_encoder: Optimizer, 
    optimizer_decoder: Optimizer,
    epoch: int, 
    outpath: Union[Path, str], 
    is_train: bool = True, 
    device: Optional[torch.device] = None
):

    if is_train:
        assert (optimizer_encoder is not None) and (optimizer_decoder is not None), "Please specify the optimizers."
        encoder.train()
        decoder.train()
        encoder_weight_path = make_dir(osp.join(outpath, "weights_encoder"))
        decoder_weight_path = make_dir(osp.join(outpath, "weights_decoder"))
    else:
        encoder.eval()
        decoder.eval()

    target_data = []
    recons_data = []
    epoch_total_loss = 0

    for i, data in enumerate(tqdm(loader)):
        p4_target = data.to(args.dtype)
        p4_recons = decoder(
            encoder(p4_target, metric=args.encoder_metric),
            metric=args.decoder_metric
        )
        recons_data.append(p4_recons.cpu().detach())

        if device is not None:
            p4_target = p4_target.to(device=device)
        target_data.append(p4_target.cpu().detach())

        batch_loss = get_loss(args, p4_recons, p4_target.to(args.dtype))
        epoch_total_loss += batch_loss.cpu().item()

        # Backward propagation
        if is_train:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            batch_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            if ('emd' in args.loss_choice.lower()) and ((i % args.save_freq) == 0 and i > 0):
                torch.save(
                    encoder.state_dict(),
                    osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth")
                )
                torch.save(
                    decoder.state_dict(),
                    osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth")
                )

    recons_data = torch.cat(recons_data, dim=0)
    target_data = torch.cat(target_data, dim=0)

    epoch_avg_loss = epoch_total_loss / len(loader)

    # Save weights
    if is_train:
        torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch}_encoder_weights.pth"))
        torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch}_decoder_weights.pth"))

    return epoch_avg_loss, recons_data, target_data


@torch.no_grad()
def validate(
    args: Namespace, 
    loader: DataLoader, 
    encoder: Encoder, 
    decoder: Decoder, 
    epoch: int, 
    outpath: Union[Path, str], 
    device: Optional[torch.device] = None
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        epoch_avg_loss, recons_data, target_data = train(
            args, loader=loader, encoder=encoder, decoder=decoder,
            optimizer_encoder=None, optimizer_decoder=None,
            epoch=epoch, outpath=outpath, is_train=False, device=device
        )
    return epoch_avg_loss, recons_data, target_data


def train_loop(
    args: Namespace, 
    train_loader: DataLoader, 
    valid_loader: DataLoader, 
    encoder: Encoder, 
    decoder: Decoder,
    optimizer_encoder: Optimizer, 
    optimizer_decoder: Optimizer, 
    outpath: Union[Path, str],
    device: torch.device = None
) -> int:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert (args.save_dir is not None), "Please specify directory of saving the models!"
    make_dir(args.save_dir)

    train_avg_losses = []
    valid_avg_losses = []
    dts = []

    best_epoch = 1
    num_stale_epochs = 0
    best_loss = math.inf
    best_epoch_dict = {"best_epoch": best_epoch, "best_loss": best_loss}
    
    if args.load_to_train:
        try:
            best_epoch_dict = torch.load(osp.join(outpath, "trained_info.pt"))
            best_epoch = best_epoch_dict["best_epoch"]
            best_loss = best_epoch_dict["best_loss"]
        except FileNotFoundError:
            pass

    outpath_train_jet_plots = make_dir(osp.join(outpath, 'jet_plots/train'))
    outpath_valid_jet_plots = make_dir(osp.join(outpath, 'jet_plots/valid'))

    total_epoch = args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch
    
    path_eval = Path(outpath) / 'model_evaluations'
    path_eval.mkdir(exist_ok=True, parents=True)

    for ep in range(args.num_epochs):
        epoch = args.load_epoch + ep if args.load_to_train else ep

        # Training
        start = time.time()
        train_avg_loss, train_recons, train_target = train(
            args, train_loader, encoder, decoder,
            optimizer_encoder, optimizer_decoder, epoch,
            outpath, is_train=True, device=device
        )
        # Validation
        valid_avg_loss, valid_recons, valid_target = validate(
            args, valid_loader, encoder, decoder,
            epoch, outpath, device=device
        )

        if (abs(valid_avg_loss) < best_loss):
            best_loss = valid_avg_loss
            num_stale_epochs = 0
            best_epoch = epoch + 1
            torch.save(
                encoder.state_dict(),
                osp.join(outpath, "weights_encoder/best_encoder_weights.pth")
            )
            torch.save(
                decoder.state_dict(),
                osp.join(outpath, "weights_decoder/best_decoder_weights.pth")
            )
            best_epoch_dict = {"best_epoch": best_epoch, "best_loss": best_loss}
            torch.save(best_epoch_dict, osp.join(outpath, "trained_info.pt"))
        else:
            num_stale_epochs += 1

        dt = time.time() - start

        if (args.abs_coord and (args.unit.lower() == 'tev')) and not args.normalized:
            # Convert to GeV for plotting
            train_target *= 1000
            train_recons *= 1000
            valid_target *= 1000
            valid_recons *= 1000

        # EMD: Plot every epoch because model trains slowly with the EMD loss.
        # Others (MSE and chamfer losses): Plot every args.plot_freq epoch or the best epoch.
        is_emd = 'emd' in args.loss_choice.lower()
        if args.plot_freq > 0:
            if (epoch >= args.plot_start_epoch):
                plot_epoch = ((epoch + 1) % args.plot_freq == 0) or (num_stale_epochs == 0)
            else:
                plot_epoch = False
        else:
            plot_epoch = (num_stale_epochs == 0)
        to_plot = is_emd or plot_epoch

        if to_plot:
            for target, recons, path in zip(
                (train_target, valid_target),
                (train_recons, valid_recons),
                (outpath_train_jet_plots, outpath_valid_jet_plots)
            ):
                logging.debug("plotting")
                plot_p(args, p4_target=target, p4_recons=recons, save_dir=path, epoch=epoch, show=False)

        dts.append(dt)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(valid_avg_loss)
        np.savetxt(path_eval / 'losses_training.txt', train_avg_losses)
        np.savetxt(path_eval / 'losses_validation.txt', valid_avg_losses)
        np.savetxt(path_eval / 'dts.txt', dts)

        logging.info(
            f'epoch={epoch+1}/{total_epoch}, train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, '
            f'{dt=}s, {num_stale_epochs=}, {best_epoch=}'
        )

        if args.plot_freq > 0:
            if (epoch > 0) and (epoch % int(args.plot_freq) == 0):
                plot_eval_results(
                    args, data=(train_avg_losses, valid_avg_losses),
                    data_name='Losses', outpath=outpath, start=epoch-args.plot_freq
                )

        if num_stale_epochs > args.patience:
            logging.info(
                f'Number of stale epochs reached the set patience ({args.patience}). Training breaks.'
            )
            return best_epoch

        if abs(valid_avg_loss) > BLOW_UP_THRESHOLD:
            logging.error('Loss blows up. Training breaks.')
            return best_epoch


    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                      data_name='Losses', outpath=outpath)
    plot_eval_results(args, data=dts, data_name='Time durations',
                      outpath=outpath)

    return best_epoch


def get_loss(
    args: Namespace, 
    p4_recons: torch.Tensor, 
    p4_target: torch.Tensor,
    encoder: Encoder = None,
    decoder: Decoder = None
) -> torch.Tensor:
    if args.loss_choice.lower() in ['chamfer', 'chamferloss', 'chamfer_loss']:
        from utils.losses import ChamferLoss
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = chamferloss(p4_recons, p4_target, jet_features_weight=args.chamfer_jet_features_weight)  # output, target

    if args.loss_choice.lower() in ['emd', 'emdloss', 'emd_loss']:
        from utils.losses import EMDLoss
        emdloss = EMDLoss(
            num_particles=p4_recons.shape[-2],
            device='cuda' if 'cuda' in str(args.device).lower() else 'cpu'
        )
        batch_loss = emdloss(p4_target, p4_recons)  # true, output

    if args.loss_choice.lower() in ['mse', 'mseloss', 'mse_loss']:
        mseloss = nn.MSELoss()
        batch_loss = mseloss(p4_recons, p4_target)  # output, target

    if args.loss_choice.lower() in ['hybrid', 'combined', 'mix']:
        from utils.losses import ChamferLoss
        from utils.losses import EMDLoss
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        emdloss = EMDLoss(
            num_particles=p4_recons.shape[-2],
            device='cuda' if 'cuda' in str(args.device).lower() else 'cpu'
        )
        batch_loss = args.chamfer_loss_weight * chamferloss(
            p4_recons, p4_target, 
            jet_features_weight=args.chamfer_jet_features_weight
        ) + emdloss(
            p4_target, p4_recons
        )
    
    # regularizations
    if (encoder is not None) and (decoder is not None):
        if args.l1_lambda > 0:
            batch_loss = batch_loss + args.l1_lambda * (encoder.l1_norm() + decoder.l1_norm())
        if args.l2_lambda > 0:
            batch_loss = batch_loss + args.l2_lambda * (encoder.l2_norm() + decoder.l2_norm())
    return batch_loss
