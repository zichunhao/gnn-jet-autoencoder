import os.path as osp
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from utils.argparse_utils import get_bool, get_device, get_dtype
from utils.argparse_utils import (
    parse_model_settings,
    parse_eval_settings
)
from utils.jet_analysis import plot_p, get_ROC_AUC, anomaly_scores_sig_bkg
from utils.initialize import initialize_models, initialize_test_dataloader
from utils.permutation import PermutationTest
from utils.utils import make_dir, get_best_epoch
from utils.train import validate


def test(args):
    test_loader = initialize_test_dataloader(
        paths=args.test_data_paths, batch_size=args.test_batch_size
    )

    # Load models
    encoder, decoder = initialize_models(args)
    
    # permutation test
    permutation_test = PermutationTest(
        encoder=encoder, 
        decoder=decoder,
        device=args.device,
        dtype=args.dtype
    )
    perm_result = permutation_test(test_loader, verbose=False)
    logging.info(f"Permutation invariance: {perm_result['invariance']}")
    logging.info(f"Permutation equivariance: {perm_result['equivariance']}")
    

    _, recons, target, latent = validate(
        args,
        test_loader,
        encoder,
        decoder,
        args.load_epoch,
        args.load_path,
        args.device,
    )
    test_path = make_dir(
        osp.join(args.load_path, f"test_{args.jet_type}_jets_{args.load_epoch}")
    )
    torch.save(target, osp.join(test_path, "target.pt"))
    torch.save(recons, osp.join(test_path, "reconstructed.pt"))
    torch.save(latent, osp.join(test_path, "latent.pt"))
    logging.info(f"Data saved exported to {test_path}.")

    fig_path = make_dir(osp.join(test_path, "jet_plots"))
    if args.abs_coord and (args.unit.lower() == "tev"):
        # Convert to GeV for plotting
        scale = 1000
    else:
        scale = 1

    jet_images_same_norm, jet_images = plot_p(
        args, 
        target*scale, 
        recons*scale, 
        fig_path, 
        jet_type=args.jet_type
    )
    torch.save(jet_images_same_norm, osp.join(test_path, "jet_images_same_norm.pt"))
    torch.save(jet_images, osp.join(test_path, "jet_images.pt"))
    logging.info("Plots finished.")

    # anomaly detection
    if (args.anomaly_detection) and (len(args.signal_paths) > 0):
        logging.info(f"Anomaly detection started. Signal paths: {args.signal_paths}")
        path_ad = Path(make_dir(osp.join(test_path, "anomaly_detection")))
        bkg_recons, bkg_target = recons, target

        torch.save(bkg_recons, path_ad / f"{args.jet_type}_recons.pt")
        torch.save(bkg_target, path_ad / f"{args.jet_type}_target.pt")
        torch.save(latent, path_ad / f"{args.jet_type}_latent.pt")

        sig_recons_list = []
        sig_target_list = []
        sig_scores_list = []

        # background vs single signal
        for signal_path, signal_type in zip(args.signal_paths, args.signal_types):
            logging.info(f"Anomaly detection: {args.jet_type} vs {signal_type}.")
            path_ad_single = path_ad / f"single_signals/{signal_type}"
            sig_loader = initialize_test_dataloader(
                paths=signal_path, batch_size=args.test_batch_size
            )
            _, sig_recons, sig_target, sig_latent = validate(
                args,
                sig_loader,
                encoder,
                decoder,
                args.load_epoch,
                args.load_path,
                args.device
            )

            scores_dict, true_labels, sig_scores, bkg_scores = anomaly_scores_sig_bkg(
                sig_recons,
                sig_target,
                bkg_recons,
                bkg_target,
                include_emd=True,
                polar_coord=args.polar_coord,
                abs_coord=args.abs_coord,
                batch_size=args.test_batch_size,
            )
            get_ROC_AUC(scores_dict, true_labels, save_path=path_ad_single)
            plot_p(
                args,
                sig_target * scale,
                sig_recons * scale,
                save_dir=path_ad_single,
                jet_type=signal_type,
            )

            # add to list
            sig_recons_list.append(sig_recons)
            sig_target_list.append(sig_target)
            sig_scores_list.append(sig_scores)

            # save results
            torch.save(sig_recons, path_ad_single / f"{signal_type}_recons.pt")
            torch.save(sig_target, path_ad_single / f"{signal_type}_target.pt")
            torch.save(sig_latent, path_ad_single / f"{signal_type}_latent.pt")

        # background vs. all signals
        logging.info(f"Anomaly detection: {args.jet_type} vs {args.signal_types}.")
        sig_recons = torch.cat(sig_recons_list, dim=0)
        sig_target = torch.cat(sig_target_list, dim=0)

        # concatenate all signal scores
        sig_scores = {
            k: np.concatenate([v[k] for v in sig_scores_list], axis=0)
            for k in sig_scores_list[0].keys()
        }
        # signals and backgrounds
        scores_dict = {
            k: np.concatenate([sig_scores[k], bkg_scores[k]]) for k in sig_scores.keys()
        }
        true_labels = np.concatenate(
            [
                np.ones_like(sig_scores[list(sig_scores.keys())[0]]),
                -np.ones_like(bkg_scores[list(sig_scores.keys())[0]]),
            ]
        )
        get_ROC_AUC(scores_dict, true_labels, save_path=path_ad)

    elif (args.anomaly_detection) and (len(args.signal_paths) > 0):
        logging.error("No signal paths given for anomaly detection.")


def setup_argparse():
    parser = argparse.ArgumentParser(description="GNN Autoencoder on Test Dataset")

    # Model
    parse_model_settings(parser)
    
    # Data
    parser.add_argument(
        '--test-data-paths', type=str, nargs='+', 
        help='Paths of the test data.'
    )
    parser.add_argument(
        '-j', '--jet-type', type=str, default='qcd',
        help="Jet type to train. Options: ('qcd', 'g', 'q', 't', 'w', 'z')."
    )
    parser.add_argument(
        '-tbs', '--test-batch-size', type=int, default=128, metavar='',
        help='Test batch size.'
    )
    parser.add_argument(
        '--unit', type=str, default='TeV',
        help="The unit of momenta. Choices: ('GeV', 'TeV'). Default: TeV. "
    )
    parser.add_argument(
        '--abs-coord', type=get_bool, default=False, metavar='',
        help='Whether the data is in absolute coordinates. False when relative coordinates are used.'
    )
    parser.add_argument(
        '--polar-coord', type=get_bool, default=True, metavar='',
        help='Whether the data is in polar coordinates (pt, eta, phi). False when Cartesian coordinates are used.'
    )
    parser.add_argument(
        '--normalized', type=get_bool, default=False, metavar='',
        help='Whether the data is normalized. False when unnormalized data is used.'
    )
    
    parser.add_argument(
        '--device', type=get_device, default=get_device('-1'), metavar='',
        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1'). "
        "Default: -1, which means deciding device based on whether gpu is available."
    )
    parser.add_argument(
        '--dtype', type=get_dtype, default=torch.float64, metavar='',
        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: torch.float64"
    )

    

    # Test
    parser.add_argument(
        '--load-path', type=str, required=True, metavar='',
        help='Path of the trained model to load.'
    )
    parser.add_argument(
        '--load-epoch', type=int, default=-1, metavar='',
        help='Epoch number of the trained model to load.'
    )
    parser.add_argument(
        '--loss-choice', type=str, default='ChamferLoss', metavar='',
        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')"
    )
    parser.add_argument(
        '--loss-norm-choice', type=str, default='cartesian', metavar='',
        help="Choice of calculating the norms of 4-vectors when calculating the loss. "
        "Options: ['cartesian', 'minkowskian', 'polar']. \n"
        "'cartesian': (+, +, +, +). \n"
        "'minkowskian': (+, -, -, -) \n"
        "'polar': convert to (E, pt, eta, phi) paired with metric (+, +, +, +) \n"
        "Default: 'cartesian.'"
    )
    parser.add_argument(
        '--chamfer-jet-features-weight', type=float, default=1,
        help="The weight of jet momenta when adding to the particle momenta chamfer loss."
    )
    parser.add_argument(
        "--chamfer-jet-features",
        type=get_bool,
        default=True,
        help="Whether to take into the jet features.",
    )
    # Plots
    parse_eval_settings(parser)

    # Anomaly detection
    parser.add_argument(
        "--anomaly-detection",
        action="store_true",
        default=False,
        help="Whether to run anomaly detection.",
    )
    parser.add_argument(
        "--anomaly-scores-batch-size",
        type=int,
        default=-1,
        metavar="",
        help="Batch size when computing anomaly scores. Used for calculating chamfer distances. "
        "Default: -1, which means not using batch size.",
    )
    parser.add_argument(
        "--signal-paths",
        nargs="+",
        type=str,
        metavar="",
        default=[],
        help="Paths to all signal files",
    )
    parser.add_argument(
        "--signal-types",
        nargs="+",
        type=str,
        metavar="",
        default=[],
        help="Types of jets in the signal files",
    )
    parser.add_argument(
        "--plot-num-rocs",
        type=int,
        metavar="",
        default=-1,
        help="Number of ROC curves to keep when plotting (after sorted by AUC). "
        "If the value takes one of {0, -1}, all ROC curves are kept.",
    )

    args = parser.parse_args()
    args.load_to_train = True
    if args.load_epoch < 0:
        args.load_epoch = get_best_epoch(args.load_path, num=args.load_epoch)
    if args.load_path is None:
        raise ValueError("--model-path needs to be specified.")
    args.l1_lambda = args.l2_lambda = 0
    return args


if __name__ == "__main__":
    import sys

    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    logging.info(f"{args=}")
    test(args)
