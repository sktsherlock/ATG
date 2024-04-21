import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusions.MultiBench import *


def add_common_args(argparser):
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--n-runs", type=int, default=3, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
    )
    argparser.add_argument(
        "--lr", type=float, default=0.005, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=256, help="number of hidden units"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--min-lr", type=float, default=0.0001, help="the min learning rate"
    )
    argparser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="the smoothing factor"
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    # ! default
    argparser.add_argument(
        "--log-every", type=int, default=20, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--eval_steps", type=int, default=1, help="eval in every epochs"
    )
    argparser.add_argument(
        "--early_stop_patience", type=int, default=None, help="when to stop the  training loop to be aviod of the overfiting"
    )
    argparser.add_argument(
        "--warmup_epochs", type=int, default=None, help="The warmup epochs"
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="The seed for the teacher models"
    )
    # ! Model related
    argparser.add_argument(
        "--model_name", type=str, default='GCN', help="The model name.",
    )
    argparser.add_argument(
        "--fusion_name", type=str, default='LRTF', help="The fusion model name.",
    )
    # ! Data related
    argparser.add_argument(
        "--data_name", type=str, default=None, help="The dataset name.",
    )
    argparser.add_argument(
        "--feature", type=str, default=None, help="Use Unimodal feature embedding as feature",
    )
    argparser.add_argument(
        "--text_feature", type=str, default=None, help="Use text feature embedding as feature",
    )
    argparser.add_argument(
        "--visual_feature", type=str, default=None, help="Use visual feature embedding as feature",
    )
    argparser.add_argument(
        "--graph_path", type=str, default=None, help="The datasets to be implemented."
    )
    argparser.add_argument(
        "--undirected", type=bool, default=True, help="Whether to undirect the graph."
    )
    argparser.add_argument(
        "--selfloop", type=bool, default=True, help="Whether to add self loop in the graph."
    )
    argparser.add_argument(
        "--metric", type=str, default='accuracy', choices=['accuracy', 'precision', 'recall', 'f1'],
        help="The metric to be used."
    )
    argparser.add_argument(
        "--average", type=str, default='weighted', choices=['weighted', 'micro', 'macro', None]
    )
    # ! Split datasets
    argparser.add_argument(
        "--inductive", type=bool, default=False, help="Whether to do inductive learning experiments."
    )
    argparser.add_argument(
        "--train_ratio", type=float, default=0.6, help="training ratio"
    )
    argparser.add_argument(
        "--val_ratio", type=float, default=0.2, help="training ratio"
    )
    argparser.add_argument(
        "--fewshots", type=int, default=None, help="fewshots values"
    )


def add_fusion_args(argparser):
    argparser.add_argument("--input_dims", nargs="+", type=int, help="Input dimensions of the modalities")
    argparser.add_argument(
        "--output_dim", type=int, default=768, help="number of output_dim"
    )
    argparser.add_argument(
        "--rank", type=int, default=10, help="a hyperparameter of LRTF. See link above for details"
    )


def add_gat_args(argparser):
    argparser.add_argument(
        "--no-attn-dst", type=bool, default=True, help="Don't use attn_dst."
    )
    argparser.add_argument(
        "--n-heads", type=int, default=3, help="number of heads"
    )
    argparser.add_argument(
        "--attn-drop", type=float, default=0.0, help="attention drop rate"
    )
    argparser.add_argument(
        "--edge-drop", type=float, default=0.0, help="edge drop rate"
    )


def add_sage_args(argparser):
    argparser.add_argument(
        "--aggregator", type=str, default="mean", choices=["mean", "gcn", "pool", "lstm"], help="Specify the aggregator option"
    )


def add_sgc_args(argparser):
    argparser.add_argument(
        "--bias", type=bool, default=True, help="control the SGC bias."
    )
    argparser.add_argument(
        "--k", type=int, default=2, help="number of k"
    )


def add_revgat_args(argparser):
    argparser.add_argument(
        "--use_symmetric_norm", type=bool, default=True, help="if False, no W."
    )


def add_appnp_args(argparser):
    argparser.add_argument(
        "--edge_drop", type=float, default=0.5, help="edge propagation dropout"
    )
    argparser.add_argument(
        "--k_ps", type=int, default=10, help="Number of propagation steps"
    )
    argparser.add_argument(
        "--alpha", type=float, default=0.1, help="Teleport Probability"
    )
    argparser.add_argument(
        "--input_dropout", type=float, default=0.5, help="dropout rate"
    )


def gen_model(args, device):
    if args.model_name is 'GCN':
        model = GCN(args.output_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)
    else:
        raise ValueError('GNN must be in the implementrary.')
    return model


def gen_fusion(args, device):
    if args.fusion_name is 'TF':
        fusion = TensorFusion()
    elif args.fusion_name is 'LRTF':
        fusion = LowRankTensorFusion(args.input_dims, args.output_dim, args.rank).to(device)
    else:
        raise ValueError('Fusion must be in the implementrary.')
    return fusion

