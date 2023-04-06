import logging
import os
import os.path as osp
from pathlib import Path
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np


def get_model_folder(args):
    make_dir(args.save_dir)
    path = Path(args.save_dir) / get_model_fname(args)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_fname(args):
    model_fname = f"GNNAutoencoder_{args.jet_type}Jet_LatentDim{args.latent_node_size}_LatentMap_{args.latent_map.replace(' ', '')}"
    return model_fname


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_p_polar(p, eps=1e-16, keep_p0=False):
    """
    (E, px, py, pz) -> (eta, phi, pt) or (E, eta, phi, pt)

    keep_p0: bool
        Whether to keep p0.
        Optional, default: False
    """
    if p.shape[-1] == 4:  # 4-vectors
        px = p[..., 1]
        py = p[..., 2]
        pz = p[..., 3]
        get_E = lambda p: p[..., 0]
    elif p.shape[-1] == 3:  # 3-vectors
        px = p[..., 0]
        py = p[..., 1]
        pz = p[..., 2]
        get_E = lambda p: torch.sum(torch.pow(p, 2), axis=-1)
    else:
        raise ValueError(
            f"Invalid dimension of p; p should be 3- or 4-vectors. Found: {p.shape[-1]=}."
        )

    pt = torch.sqrt(px**2 + py**2 + eps)
    try:
        eta = torch.asinh(pz / (pt + eps))
    except AttributeError:
        eta = arcsinh(pz / (pt + eps))
    phi = torch.atan2(py + eps, px + eps)

    if not keep_p0:
        return torch.stack((eta, phi, pt), dim=-1)
    else:
        E = get_E(p)
        return torch.stack((E, eta, phi, pt), dim=-1)


def arcsinh(z):
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def save_data(data, data_name, is_train, outpath, epoch=-1):
    """
    Save data like losses and dts. If epoch is -1, the data will be considered a global data, such as
    the losses over all epochs.
    """
    outpath = make_dir(osp.join(outpath, "model_evaluations/pt_files"))
    if isinstance(data, torch.Tensor):
        data = data.cpu()

    if is_train is None:
        if epoch >= 0:
            torch.save(data, osp.join(outpath, f"{data_name}_epoch_{epoch}.pt"))
        else:
            torch.save(data, osp.join(outpath, f"{data_name}.pt"))
        return

    if epoch >= 0:
        if is_train:
            torch.save(data, osp.join(outpath, f"train_{data_name}_epoch_{epoch}.pt"))
        else:
            torch.save(data, osp.join(outpath, f"valid_{data_name}_epoch_{epoch}.pt"))
    else:
        if is_train:
            torch.save(data, osp.join(outpath, f"train_{data_name}.pt"))
        else:
            torch.save(data, osp.join(outpath, f"valid_{data_name}.pt"))


def plot_eval_results(args, data, data_name, outpath, start=None):
    """
    Plot evaluation results
    """
    outpath = make_dir(osp.join(outpath, "model_evaluations/evaluation_plots"))
    if args.load_to_train:
        start = args.load_epoch + 1
    else:
        start = 1 if start is None else start

    # (train, label)
    if type(data) in [tuple, list] and len(data) == 2:
        train, valid = data
        x = [start + i for i in range(len(train))]
        if isinstance(train, torch.Tensor):
            train = train.detach().cpu().numpy()
        if isinstance(valid, torch.Tensor):
            valid = valid.detach().cpu().numpy()
        plt.plot(x, train, label="Train", alpha=0.8)
        plt.plot(x, valid, label="Valid", alpha=0.8)
        if (np.array(train) > 0).all() and (np.array(valid) > 0).all():
            plt.yscale("log")
        plt.legend()
    # only one type of data (e.g. dt)
    else:
        x = [start + i for i in range(len(data))]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        plt.plot(x, data)

    plt.xlabel("Epoch")
    plt.ylabel(data_name)
    plt.title(data_name)
    save_name = "_".join(data_name.lower().split(" "))
    plt.savefig(osp.join(outpath, f"{save_name}.pdf"), bbox_inches="tight")
    plt.close()


def get_best_epoch(model_path: str, num: int = -1) -> int:
    """Return the best epoch number if it is saved in the model path.
    Otherwise, return the latest epoch number.
    """
    try:
        info = torch.load(osp.join(model_path, "trained_info.pt"))
        return info["best_epoch"]
    except FileNotFoundError:
        # trained_info.pt not saved
        path = osp.join(model_path, "weights_decoder/*.pth")
        file_list = glob.glob(path)
        epochs = [
            [int(s) for s in filename.split("_") if s.isdigit()]
            for filename in file_list
        ]
        epochs.sort()
        try:
            latest = epochs[num][0]
        except IndexError:
            try:
                latest = epochs[-1][0]
            except IndexError:
                logging.warning(f"Model does not exist in {model_path}")
                return -1
        return latest
