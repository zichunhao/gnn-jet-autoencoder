import argparse
import logging
from pathlib import Path
from typing import Union
import jetnet
import numpy as np
import torch

NUM_DEV_INSTANCES = 50

POLAR_REL = ('polarrel', 'polar_rel')
POLAR_ABS = ('polar', 'polarabs', 'polar_abs')
CARTESIAN = ('cartesian', 'cart')

def prepare(
    jet_type: str,
    save_dir: Union[str, Path],
    test_portion: float = 0.2,
    normalize: bool = False,
    four_vec: bool = False
):
    logging.info(f"Downloading data ({jet_type=}) from JetNet.")
    data = jetnet.datasets.JetNet(
        jet_type=jet_type,
        data_dir=args.jetnet_dir
    )
    logging.info(f"Preparing data ({jet_type=}).")

    if isinstance(save_dir, Path):
        pass
    elif isinstance(save_dir, str):
        save_dir = Path(save_dir)
    else:
        raise TypeError(
            "save_path must be of type a str or pathlib.Path. "
            f"Got: {type(save_dir)}."
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    jet = data.jet_data
    p = data.particle_data

    # jet momenta components
    Pt, Eta, Mass = jet[..., 1], jet[..., 2], jet[..., 3]
    Phi = np.random.random(Eta.shape) * 2 * np.pi  # [0, 2pi]

    # particle momenta components (relative coordinates)
    eta_rel, phi_rel, pt_rel, mask = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    
    coord = args.coord.lower().replace(' ', '_').replace('-', '_')
    
    if coord in CARTESIAN:
        # particle momenta components (polar coordinates)
        pt = pt_rel * Pt.reshape(-1, 1)
        eta = eta_rel + Eta.reshape(-1, 1)
        phi = phi_rel + Phi.reshape(-1, 1)
        phi = ((phi + np.pi) % (2 * np.pi)) - np.pi  # [-pi, pi]
        mask = torch.from_numpy(mask)

        # Cartesian coordinates
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        if four_vec:
            m = np.random.random(eta.shape) * 1e-3  # O(1e-4 GeV)
            p0 = np.sqrt((pt * np.cosh(eta))**2 + m**2)
            p4 = torch.from_numpy(np.stack([p0, px, py, pz], axis=-1))
            p = p4 * mask.unsqueeze(-1)
        else:
            p = torch.from_numpy(np.stack([px, py, pz], axis=-1))
        if not normalize:
            p = p / 1000  # GeV -> TeV
        else:
            p = p / p4.abs().max()
    elif coord in POLAR_REL:
        p = torch.from_numpy(np.stack([pt_rel, eta_rel, phi_rel], axis=-1))
    elif coord in POLAR_ABS:
        pt = pt_rel * Pt.reshape(-1, 1)
        eta = eta_rel + Eta.reshape(-1, 1)
        phi = phi_rel + Phi.reshape(-1, 1)
        phi = ((phi + np.pi) % (2 * np.pi)) - np.pi  # [-pi, pi]
        p = torch.from_numpy(np.stack([pt, eta, phi], axis=-1))
    else:
        raise ValueError(f"Invalid coordinate system: {args.coord}")

    torch.save(p, save_dir / f"{jet_type}_jets_30p_all.pt")

    # training-test split
    split_idx = int(len(data) * (1 - test_portion))
    torch.save(p[:split_idx], save_dir / f"{jet_type}_jets_30p_train.pt")
    torch.save(p[split_idx:], save_dir / f"{jet_type}_jets_30p_test.pt")
    torch.save(p[:NUM_DEV_INSTANCES], save_dir / f"{jet_type}_jets_30p_small.pt")
    logging.info(
        f"Data saved in {save_dir} as {jet_type}_jets_30p_all.pt, {jet_type}_jets_30p_train.pt, "
        f"{jet_type}_jets_30p_test.pt, and {jet_type}_jets_30p_small.pt."
    )

    return


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # args
    parser = argparse.ArgumentParser(
        description='Prepare dataset for LGN Autoencoder'
    )
    parser.add_argument(
        '-j', '--jet_types',
        nargs='+', type=str, default=['g', 'q', 't', 'w', 'z'],
        help='List of jet types to download and preprocess.'
    )
    parser.add_argument(
        '-s', '--save-dir',
        type=str, required=True,
        help='Directory to save preprocessed data.'
    )
    parser.add_argument(
        '--jetnet-dir',
        type=str, default='data',
        help='Directory to save JetNet data.'
    )
    parser.add_argument(
        '-t', '--test-portion',
        type=float, default=0.2,
        help="Test portion of the data."
    )
    parser.add_argument(
        '--coord', type=str, default='cartesian',
        help="Coordinate system to use for the data. "
        "Options: ('cartesian', 'polar', 'polar_rel')."
    )
    parser.add_argument(
        '--normalize',
        action='store_true', default=False,
        help="If --coord is Cartesian, normalize the data by the global maximum (of the absolute value)."
    )
    parser.add_argument(
        '--four-vec',
        action='store_true', default=False,
        help="If --coord is Cartesian, use four-vector particle features. "
        "Only valid for 'cartesian' coordinate system."
    )
    
    args = parser.parse_args()
    logging.info(f"{args=}")
    
    for jet_type in args.jet_types:
        prepare(
            jet_type=jet_type,
            save_dir=Path(args.save_dir),
            test_portion=args.test_portion,
            normalize=args.normalize,
            four_vec=args.four_vec
        )
