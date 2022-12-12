import argparse
import numpy as np
import torch


def parse_data_settings(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-j",
        "--jet-type",
        type=str,
        default="qcd",
        help="Jet type to train. Options: ('qcd', 'g', 'q', 't', 'w', 'z').",
    )
    parser.add_argument("--data-paths", type=str, nargs="+", help="Paths of the data.")
    parser.add_argument(
        "--test-data-paths", type=str, nargs="+", help="Paths of the test data."
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=64, metavar="", help="Batch size."
    )
    parser.add_argument(
        "-tbs",
        "--test-batch-size",
        type=int,
        default=128,
        metavar="",
        help="Test batch size.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="TeV",
        help="The unit of momenta. Choices: ('GeV', 'TeV'). Default: TeV. ",
    )
    parser.add_argument(
        "--abs-coord",
        type=get_bool,
        default=True,
        metavar="",
        help="Whether the data is in absolute coordinates. False when relative coordinates are used.",
    )
    parser.add_argument(
        "--polar-coord",
        type=get_bool,
        default=False,
        metavar="",
        help="Whether the data is in polar coordinates (pt, eta, phi). False when Cartesian coordinates are used.",
    )
    parser.add_argument(
        "--normalized",
        type=get_bool,
        default=False,
        metavar="",
        help="Whether the data is normalized. False when unnormalized data is used.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.65,
        metavar="",
        help="The fraction (or number) of data used for training.",
    )
    return parser


def parse_model_settings(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--num-jet-particles",
        type=int,
        default=30,
        metavar="",
        help="Number of particles per jet (batch) in the input. Default: 30 for the hls4ml 30p data.",
    )
    parser.add_argument(
        "--vec-dims",
        type=int,
        default=3,
        metavar="",
        help="Dimension of vectors. Default: 4 for 4-vectors.",
    )
    parser.add_argument(
        "--latent-node-size",
        type=int,
        default=20,
        metavar="",
        help='Dimension of latent vectors. If --latent-map is "local" or "node", this stands for the size of feature vector per node.',
    )

    # encoder
    parser.add_argument(
        "--encoder-edge-sizes",
        type=get_list_of_list,
        default=[[32, 128, 64, 16]],
        metavar="",
        help="Edge convolution layer width in each message passing step in encoder. "
        "To enter a list of lists, use ';' to separate the lists and ',' to separate the elements in each list.",
    )
    parser.add_argument(
        "--encoder-node-sizes",
        type=get_list_of_list,
        default=[[16], [32], [8]],
        metavar="",
        help="Edge convolution layer width in each message passing step in encoder. "
        "To enter a list of lists, use ';' to separate the lists and ',' to separate the elements in each list.",
    )
    parser.add_argument(
        "--encoder-num-mps",
        type=int,
        default=3,
        metavar="",
        help="Number of message passing steps in encoder.",
    )
    parser.add_argument(
        "--encoder-alphas",
        type=float,
        default=0.2,
        metavar="",
        help="Alpha values for the leaky relu layers in encoder.",
    )
    parser.add_argument(
        "--encoder-dropout",
        type=float,
        default=0.0,
        metavar="",
        help="Dropout rate in encoder.",
    )
    parser.add_argument(
        "--encoder-batch-norm",
        action="store_true",
        default=False,
        help="Call to include batch normalizations in encoder. Default: True.",
    )
    parser.add_argument(
        "--encoder-metric",
        type=str,
        default="euclidean",
        metavar="",
        help="The metric for distance in encoder. Options: ('minkoskian', 'euclidean'). Default: 'euclidean'.",
    )

    parser.add_argument(
        "--latent-map",
        type=str,
        default="mean",
        metavar="",
        help="Method to map from GNN to latent space. Options: ('mean', 'max', 'min', 'local mix', 'global mix'). Default: 'mean'.",
    )

    # decoder
    parser.add_argument(
        "--decoder-edge-sizes",
        type=get_list_of_list,
        default=[[32, 128, 64, 16]],
        metavar="",
        help="Edge convolution layer width in each message passing step in decoder. "
        "To enter a list of lists, use ';' to separate the lists and ',' to separate the elements in each list.",
    )
    parser.add_argument(
        "--decoder-node-sizes",
        type=get_list_of_list,
        default=[[16], [32], [8]],
        metavar="",
        help="Edge convolution layer width in each message passing step in decoder."
        "To enter a list of lists, use ';' to separate the lists and ',' to separate the elements in each list.",
    )
    parser.add_argument(
        "--decoder-num-mps",
        type=int,
        default=3,
        metavar="",
        help="Number of message passing steps in decoder.",
    )
    parser.add_argument(
        "--decoder-alphas",
        type=float,
        default=0.2,
        metavar="",
        help="Alpha value for the leaky relu layers in decoder.",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=0.0,
        metavar="",
        help="Dropout rate in decoder.",
    )
    parser.add_argument(
        "--decoder-batch-norm",
        action="store_true",
        default=False,
        help="Whether to include batch normalizations in decoder. Default: True.",
    )
    parser.add_argument(
        "--decoder-metric",
        type=str,
        default="euclidean",
        metavar="",
        help="The metric for distance in decoder. Options: ('minkoskian', 'cartesian'). Default: 'cartesian'.",
    )
    return parser


def parse_training_settings(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        type=get_device,
        default=get_device("-1"),
        metavar="",
        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')."
        "Default: -1, which means deciding device based on whether gpu is available.",
    )
    parser.add_argument(
        "--dtype",
        type=get_dtype,
        default=torch.float64,
        metavar="",
        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: torch.float64",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="",
        help="Learning rate of the backpropagation.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        metavar="",
        help="The optimizer to use. Options:('adam', 'rmprop') Default: 'adam'",
    )
    parser.add_argument(
        "-e",
        "--num-epochs",
        type=int,
        default=64,
        metavar="",
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=-1,
        metavar="",
        help="Patience for early stopping. Use -1 for no early stopping.",
    )
    parser.add_argument(
        "--loss-choice",
        type=str,
        default="ChamferLoss",
        metavar="",
        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')",
    )
    parser.add_argument(
        "--loss-norm-choice",
        type=str,
        default="cartesian",
        metavar="",
        help="Choice of calculating the norms of 4-vectors when calculating the loss. "
        "Options: ['cartesian', 'minkowskian', 'polar']. \n"
        "'cartesian': (+, +, +, +). \n"
        "'minkowskian': (+, -, -, -) \n"
        "'polar': convert to (E, pt, eta, phi) paired with metric (+, +, +, +) \n"
        "Default: 'cartesian.'",
    )
    parser.add_argument(
        "--chamfer-jet-features-weight",
        type=float,
        default=1,
        help="The weight of jet momenta when adding to the particle momenta chamfer loss.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="standard-autoencoder-trained-models",
        metavar="",
        help="The directory to save trained models and figures.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=500,
        metavar="",
        help="How frequent the model weights are saved in each epoch (when using EMD loss). Default: 500.",
    )
    parser.add_argument(
        "--custom-suffix",
        type=str,
        default=None,
        metavar="",
        help="Custom suffix of the saving directory.",
    )

    # Loading existing models
    parser.add_argument(
        "--load-to-train",
        default=False,
        action="store_true",
        help="Whether to load existing (trained) model for training.",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        metavar="",
        help="Path of the trained model to load.",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        default=-1,
        metavar="",
        help="Epoch number of the trained model to load.",
    )

    # L1 and L2 regularizations
    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=1e-8,
        metavar="",
        help="penalty for L1 regularization.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0,
        metavar="",
        help="penalty for L2 regularization.",
    )
    return parser


def parse_eval_settings(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plot-freq",
        type=int,
        default=100,
        metavar="",
        help="How frequent to plot. Used when --loss-choice is not EMD. Default: 100.",
    )
    parser.add_argument(
        "--plot-start-epoch",
        type=int,
        default=-1,
        metavar="",
        help="The epoch to start plotting. Default: -1.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1e-7,
        metavar="",
        help="Cutoff value of (3-)momenta magnitude to be included in the histogram. Default: 1e-7.",
    )
    parser.add_argument(
        "--fill",
        default=False,
        action="store_true",
        help="Whether to plot filled histograms as well. True only if called in the command line.",
    )

    # jet images
    parser.add_argument(
        "--jet-image-npix",
        type=int,
        default=40,
        help="The number of pixels for the jet image",
    )
    parser.add_argument(
        "--jet-image-maxR", type=float, default=0.5, help="The maxR of the jet image"
    )
    parser.add_argument(
        "--jet-image-vmin", type=float, default=1e-10, help="vmin for LogNorm"
    )
    parser.add_argument(
        "--num-jet-images",
        type=int,
        default=15,
        help="Number of one-to-one jet images to plot.",
    )

    # reconstruction ranges
    parser.add_argument(
        "--custom-particle-recons-ranges",
        default=False,
        action="store_true",
        help="Whether to manually set the ranges of particle reconstruction errors. "
        "Call --custom-particle-recons-ranges to set true.",
    )
    parser.add_argument(
        "--particle-rel-err-min-cartesian",
        nargs="+",
        type=float,
        default=[-1, -1, -1],
        metavar="",
        help="xmin of histogram for particle reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--particle-rel-err-max-cartesian",
        nargs="+",
        type=float,
        default=[1, 1, 1],
        metavar="",
        help="xmax of histogram for particle reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--particle-padded-recons-min-cartesian",
        nargs="+",
        type=float,
        default=[-100, -100, -100],
        metavar="",
        help="xmin of histogram for reconstructed padded particles in Cartesian coordinates.",
    )
    parser.add_argument(
        "--particle-padded-recons-max-cartesian",
        nargs="+",
        type=float,
        default=[100, 100, 100],
        metavar="",
        help="xmax of histogram for reconstructed padded particles in Cartesian coordinates.",
    )

    parser.add_argument(
        "--particle-rel-err-min-polar",
        nargs="+",
        type=float,
        default=[-1, -1, -1],
        metavar="",
        help="xmin of histogram for particle reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--particle-rel-err-max-polar",
        nargs="+",
        type=float,
        default=[1, 1, 1],
        metavar="",
        help="xmax of histogram for particle reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--particle-padded-recons-min-polar",
        nargs="+",
        type=float,
        default=[-100, -1, -np.pi],
        metavar="",
        help="xmin of histogram for reconstructed padded particles in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--particle-padded-recons-max-polar",
        nargs="+",
        type=float,
        default=[100, 1, np.pi],
        metavar="",
        help="xmax of histogram for reconstructed padded particles in polar coordinates.",
    )

    parser.add_argument(
        "--custom-jet-recons-ranges",
        default=False,
        action="store_true",
        help="Whether to manually set the ranges of jet reconstruction errors. "
        "Call --custom-jet-recons-ranges to set true.",
    )
    parser.add_argument(
        "--jet-rel-err-min-cartesian",
        nargs="+",
        type=float,
        default=[-1, -1, -1, -1],
        metavar="",
        help="xmin of histogram for jet reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--jet-rel-err-max-cartesian",
        nargs="+",
        type=float,
        default=[1, 1, 1, 1],
        metavar="",
        help="xmax of histogram for jet reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--jet-rel-err-min-polar",
        nargs="+",
        type=float,
        default=[-1, -1, -1, -1],
        metavar="",
        help="xmin of histogram for jet reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--jet-rel-err-max-polar",
        nargs="+",
        type=float,
        default=[1, 1, 1, 1],
        metavar="",
        help="xmax of histogram for jet reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )

    return parser


def get_bool(arg):
    """
    Adapted from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("true", "t", "1"):
        return True
    elif arg.lower() in ("false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. "
            "Options: For True, anything in ('true', 't', '1') can be used. "
            "For False, anything in ('false', 'f', '0') can be used. "
            "Cases are ignored."
        )


def get_device(arg):
    if arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_dtype(arg):
    if arg is None:
        return torch.float64

    if arg.lower() == "float":
        dtype = torch.float
    elif arg.lower() == "double":
        dtype = torch.double
    else:
        dtype = torch.float64
    return dtype


def get_list_of_list(arg):
    if arg[-1] == ";":
        arg = arg[:-1]
    return [[int(item) for item in s.split(",") if s != ""] for s in arg.split(";")]
