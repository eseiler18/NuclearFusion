"""eval a model on the data set"""
import argparse

import os
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Import from src and model
from src.path import (DATA_CLEAN, LABEL_PATH, OUT_DIR,
                      generate_result_filename)
from src.dataset import NuclearFusionTimeSeriesDataset
from src.metric import confusion_matrix
from models.WaveNet import WaveNetClassifier
from src.plot_utils import plot_conf_matrix

parser = argparse.ArgumentParser(description='disruption prediction JET')
# data specific
parser.add_argument('--parquet-dir', default=DATA_CLEAN, type=str,
                    help='parquet data directory (default: DATA_CLEAN)')
parser.add_argument('--label-dir', default=LABEL_PATH, type=str,
                    help='label table directory (default: LABEL_PATH)')

# model specific
parser.add_argument('--model_name', type=str,
                    default="bestwavenet-150chan-2epoch-10062022.pt",
                    help='name model (default:)')
parser.add_argument('--model_dir', type=str, default=OUT_DIR,
                    help='directory of model (default: OUT_DIR)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='batch size (default: 1)')
parser.add_argument('--input-channels', type=int, default=150,
                    help='number of in channels (default: 373 for clean shot)')
parser.add_argument('--n-classes', type=int, default=1,
                    help='number classification classes'
                    ' (1 for binary classification) (default: 1)')
parser.add_argument('--kernel-size', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--dilation-size', type=int, default=2,
                    help='dilatation size (default: 2)')
parser.add_argument('--stack-size', type=int, default=4,
                    help='# of stack (default: 4)')
parser.add_argument('--layer-size', type=int, default=6,
                    help='# of layer (default: 6)')
parser.add_argument('--nrecept', type=int, default=507,
                    help='receptive field sequence length (default: 507)')
parser.add_argument('--nsub', type=int, default=2000,
                    help='sequence length to optimize over (default: 2000)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout value (default: 0)')

# other
parser.add_argument('--threshold-list', type=list,
                    default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] +
                    [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] +
                    [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001] +
                    [0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
                    )
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--in-memory', default=False, type=bool,
                    help='keep parquet in memory (default: True)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed (default: 0)')


def main():
    os.system('cls||clear')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print("Creating dataloader...")
    dataset = NuclearFusionTimeSeriesDataset(args.parquet_dir, args.label_dir,
                                             args.nrecept, args.nsub,
                                             in_memory=args.in_memory)
    evalLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)

    print("Loading model...")

    model = WaveNetClassifier(args.input_channels,
                              args.kernel_size,
                              args.stack_size,
                              args.layer_size,
                              args.nsub,
                              args.n_classes,
                              args.dropout)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(os.path.join(args.model_dir,
                                                      args.model_name)))
    else:
        device = torch.device("cpu")
        model.load_state_dict(torch.load(os.path.join(args.model_dir,
                                                      args.model_name),
                                         map_location=torch.device('cpu')))
    print(device)
    model.eval()
    model.to(device)

    os.system('cls||clear')

    print("Evaluate...")
    eval_loss = []
    eval_acc = []
    eval_precision = []
    eval_recall = []
    eval_f1 = []
    conf_mat = []
    threshold_list = args.threshold_list
    threshold_list.sort()
    torch.set_grad_enabled(False)
    for threshold in tqdm(threshold_list, desc='Eval threshold',
                          unit='threshold', leave=False):
        th_loss = []
        FP = 0
        FN = 0
        TP = 0
        TN = 0
        with tqdm(evalLoader, desc='eval model',
                  unit='batch', leave=False) as t1:
            for x, target, weight, ind in t1:
                x = x.to(device)
                target = target.to(device)
                weight = weight.to(device)

                output = model(x)
                output = torch.squeeze(output, dim=1)
                nrec = args.nrecept

                # calc metric
                loss = F.binary_cross_entropy(output, target[..., nrec:],
                                              weight=weight[..., nrec:])
                tp, tn, fp, fn = confusion_matrix(output, target[..., nrec:],
                                                  threshold)

                # add metric
                th_loss.append(loss)
                TP += tp
                TN += tn
                FP += fp
                FN += fn
        eval_loss.append(sum(th_loss)/len(th_loss))
        eval_acc.append((TP + TN)/(TP + FN + TN + FP))
        eval_precision.append(TP / (FP + TP))
        eval_recall.append(TP / (FN + TP))
        eval_f1.append(2*eval_precision[-1] * eval_recall[-1]/(eval_precision[-1] + eval_recall[-1]))
        conf_mat.append([TP, TN, FP, FN])

    best_threshold_index = eval_f1.index(max(eval_f1))
    best_threshold = threshold_list[best_threshold_index]
    best_acc = eval_acc[best_threshold_index]
    best_f1 = eval_f1[best_threshold_index]
    best_loss = eval_loss[best_threshold_index]
    best_confmat = conf_mat[best_threshold_index]

    print(f"Model {args.model_name} result:")
    print(f"Threshold = {best_threshold}")
    print(f"Loss = {best_loss}")
    print(f"Accuracy = {best_acc}")
    print(f"F1 score = {best_f1}")
    print(f"conf mat = {best_confmat}")
    plot_conf_matrix(best_confmat[0], best_confmat[1], best_confmat[2],
                     best_confmat[3])

    filename = generate_result_filename(args.model_name)
    history = dict()
    history["threshold"] = threshold_list
    history["eval_loss"] = eval_loss
    history["eval_acc"] = eval_acc
    history["eval_f1"] = eval_f1
    history["conf_mat"] = conf_mat

    with open(filename, 'wb') as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
