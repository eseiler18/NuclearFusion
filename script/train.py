"""Trainer"""
import argparse

import os
import pickle
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Import from src and model
from src.path import (DATA_CLEAN, LABEL_PATH, OUT_DIR, generate_model_filename,
                      generate_log_filename)
from src.dataset import NuclearFusionTimeSeriesDataset
from src.metric import metric_thresholds, confusion_matrix
from models.WaveNet import WaveNetClassifier
from src.plot_utils import plot_training

parser = argparse.ArgumentParser(description='disruption prediction JET')
# data specific
parser.add_argument('--parquet-dir', default=DATA_CLEAN, type=str,
                    help='parquet data directory (default: DATA_CLEAN)')
parser.add_argument('--label-dir', default=LABEL_PATH, type=str,
                    help='label table directory (default: LABEL_PATH)')
# model specific
parser.add_argument('--model', type=str, default="wavenet",
                    help='name model (default: "wavenet")')
parser.add_argument('--batch', type=int, default=4, metavar='N',
                    help='batch size (default: 1)')
parser.add_argument('--input-channels', type=int, default=150,
                    help='number of in channels (default: 373 for clean shot)')
parser.add_argument('--n-classes', type=int, default=1,
                    help='number classification classes'
                    ' (1 for binary classification) (default: 1)')
parser.add_argument('--kernel-size', type=int, default=3,
                    help='kernel size (default: 3)')
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
# retrain model
parser.add_argument('--prtrain-model', type=str,
                    default=None,
                    help='(default: None)')
parser.add_argument('--prtrain-log', type=str,
                    default=None,
                    help='file log training (default: None)')
parser.add_argument('--prtrain-dir', type=str, default=OUT_DIR,
                    help='directory of pretrained model')
# learning
parser.add_argument('--thresholds', type=list,
                    default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                            #, 0.05, 0.01, 0.007, 0.005,
                            #  0.002, 0.001, 0.0007, 0.0005, 0.0001, 0.00007,
                            #  0.00005, 0.00002, 0.00001],
                    help='thresholds for classification (default: )')
parser.add_argument('--split', type=float, default=0.2,
                    help='test/train split rate (default: 0.2)')
parser.add_argument('--validation', default=False, type=bool,
                    help="do validation or not")
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--label-balance', type=str, default='const',
                    help="Type of label balancing. (default: const)")
# other
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--plot', default=True, type=bool,
                    help='plot training (default: True)')
parser.add_argument('--in-memory', default=True, type=bool,
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
    if args.validation:
        trainNumbers = int(len(dataset)*(1-args.split))
        dataNumbers = [trainNumbers, len(dataset)-trainNumbers]
        trainDataset, testDataset = torch.utils.data.random_split(dataset,
                                                                  dataNumbers)
        trainDataLoader = torch.utils.data.DataLoader(trainDataset,
                                                      batch_size=args.batch,
                                                      shuffle=True)
        testDataLoader = torch.utils.data.DataLoader(testDataset,
                                                     batch_size=args.batch,
                                                     shuffle=True)
    else:
        trainDataLoader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=args.batch,
                                                      shuffle=True)

    print("Creating model...")
    model = WaveNetClassifier(args.input_channels,
                              args.kernel_size,
                              args.stack_size,
                              args.layer_size,
                              args.nsub,
                              args.n_classes,
                              args.dropout)

    if args.prtrain_model is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(os.path.join(args.prtrain_dir,
                                                          args.prtrain_model)))
        else:
            model.load_state_dict(torch.load(os.path.join(args.prtrain_dir,
                                                          args.prtrain_model),
                                             map_location=torch.device('cpu')))

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 600], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.1^(1/2))
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, patience=5)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    model.train()
    model.to(device)

    print("Training...")
    train_loss = []
    test_loss = []
    test_acc = []
    test_f1 = []
    test_threshold = []
    conf_mat = []
    epochs = args.epochs
    max_f1 = 0

    if args.prtrain_log is not None:
        with open(os.path.join(args.prtrain_dir,
                               args.prtrain_log), 'rb') as file:
            history = pickle.load(file)
            train_loss = history["train_loss"]
            test_loss = history["test_loss"]
            test_acc = history["test_acc"]
            test_f1 = history["test_f1"]
            test_threshold = history["test_threshold"]
            conf_mat = history["conf_mat"]
            max_f1 = max(test_f1)

    start_epoch = len(train_loss)
    final_epoch = start_epoch + epochs

    with trange(1 + start_epoch, final_epoch + 1,
                desc='Training', unit='epoch') as t:
        for epoch in t:
            losses = []
            accs = []
            f1s = []
            threshold = []
            FP = 0
            FN = 0
            TP = 0
            TN = 0
            # with tqdm(trainDataLoader,
            #           desc=f'Train epoch {epoch}',
            #           unit='batch', leave=False) as t1:
                # for x_train, y_train, weight, ind in t1:
            if True:
                for x_train, y_train, weight, ind in trainDataLoader:
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                    weight = weight.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward + backward + optimize
                    output = model(x_train)
                    output = torch.squeeze(output, dim=1)
                    nrec = args.nrecept
                    loss = F.binary_cross_entropy(output,
                                                  y_train[..., nrec:],
                                                  weight=weight[..., nrec:])
                    loss.backward()
                    optimizer.step()

                    # Append loss
                    losses.append(loss)
                    if not args.validation:
                        acc, f1, th = metric_thresholds(output,
                                                        y_train[..., nrec:],
                                                        args.thresholds)
                        tp, tn, fp, fn = confusion_matrix(output,
                                                          y_train[..., nrec:],
                                                          th)

                        accs.append(acc)
                        if f1 != -1000:
                            f1s.append(f1)
                            threshold.append(th)
                        TP += tp
                        TN += tn
                        FP += fp
                        FN += fn

                train_loss.append((sum(losses)/float(len(losses))).item())
                if not args.validation:
                    test_threshold.append(sum(threshold)/float(len(threshold)))
                    test_f1.append(sum(f1s)/float(len(f1s)))
                    test_acc.append(sum(accs)/float(len(accs)))
                    conf_mat.append([TP, TN, FP, FN])
                    if test_f1[-1] > max_f1:
                        max_f1 = test_f1[-1]
                        filename = generate_model_filename("best"+args.model, final_epoch, args.input_channels)
                        weights = model.state_dict()
                        torch.save(weights, filename)
                scheduler.step()
                # scheduler_plateau.step(train_loss[-1])
                if epoch%5==0:
                    print(test_f1[-1])

            if args.validation:
                with torch.no_grad():
                    losses = []
                    accs = []
                    f1s = []
                    threshold = []
                    FP = 0
                    FN = 0
                    TP = 0
                    TN = 0
                    # with tqdm(testDataLoader,
                    #           desc=f'Test epoch {epoch}',
                    #           unit='batch', leave=False) as t1:
                        # for x_test, y_test, w, _ in t1:
                    if True:
                        for x_test, y_test, w, _ in testDataLoader:
                            x_test = x_test.to(device)
                            y_test = y_test.to(device)
                            w = w.to(device)

                            output = model(x_test)
                            output = torch.squeeze(output, dim=1)
                            i = args.nrecept
                            loss = F.binary_cross_entropy(output,
                                                          y_test[..., i:],
                                                          weight=w[..., i:])
                            y_test = torch.squeeze(y_test, dim=0)
                            output = torch.squeeze(output, dim=0)

                            # calculate metrics
                            acc, f1, th = metric_thresholds(output,
                                                            y_test[..., i:],
                                                            args.thresholds)
                            tp, tn, fp, fn = confusion_matrix(output,
                                                              y_test[..., i:],
                                                              th)

                            losses.append(loss)
                            accs.append(acc)
                            if f1 != -1000:
                                f1s.append(f1)
                                threshold.append(th)
                            TP += tp
                            TN += tn
                            FP += fp
                            FN += fn

                    test_loss.append((sum(losses)/float(len(losses))).item())
                    test_threshold.append(sum(threshold)/float(len(threshold)))
                    test_f1.append(sum(f1s)/float(len(f1s)))
                    test_acc.append(sum(accs)/float(len(accs)))
                    conf_mat.append([TP, TN, FP, FN])
                    if test_f1[-1] > max_f1:
                        max_f1 = test_f1[-1]
                        filename = generate_model_filename("best"+args.model,
                                                           final_epoch, args.input_channels)
                        weights = model.state_dict()
                        torch.save(weights, filename)

    # Save history and model
    # save model
    filename = generate_model_filename(args.model, final_epoch, args.input_channels)
    weights = model.state_dict()
    torch.save(weights, filename)

    # save losses
    filename = generate_log_filename(final_epoch, args.input_channels)
    history = dict()
    history["train_loss"] = train_loss
    history["test_acc"] = test_acc
    history["test_loss"] = test_loss
    history["test_f1"] = test_f1
    history["test_threshold"] = test_threshold
    history["conf_mat"] = conf_mat

    with open(filename, 'wb') as f:
        pickle.dump(history, f)

    # plot
    if args.plot:
        plot_training(history, True)


if __name__ == "__main__":
    main()
