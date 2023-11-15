import torch
from Utils.utils import parse_args
from Trainer.train import run_exp
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_exp(args.attribute, args.task, args.model_name, args.data_name, args)
