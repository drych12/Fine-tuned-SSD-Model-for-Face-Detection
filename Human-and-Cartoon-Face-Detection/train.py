"""
This code is the main training code.
"""
import argparse
import itertools
import logging
import os
import sys
import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

from models.datasets.voc_dataset import VOCDataset
from models.datasets.my_dataset import MyDataset
from models.nn.multibox_loss import MultiboxLoss
from models.ssd.config.fd_config import define_img_size
from models.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from models.ssd.config import fd_config
from models.ssd.data_preprocessing import TrainAugmentation, TestTransform
from models.ssd.ssd import MatchPrior

def lr_poly(base_lr, iter):
    return base_lr * ((1 - float(iter) / args.num_epochs) ** (args.power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.lr, i_iter)
    optimizer.param_groups[0]['lr'] = lr


def train(loader, net, criterion, optimizer, device, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    total = len(loader)
    with tqdm.tqdm(enumerate(loader), total=total) as tbar:
        for i, data in tbar:
            # print(".", end="", flush=True)
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            tbar.set_description(f"loss: {loss.item():.4f}")
            
    avg_loss = running_loss / total
    avg_reg_loss = running_regression_loss / total
    avg_clf_loss = running_classification_loss / total
    logging.info(
        f"Epoch: {epoch}, Step: {i}, " +
        f"Average Loss: {avg_loss:.4f}, " +
        f"Average Regression Loss {avg_reg_loss:.4f}, " +
        f"Average Classification Loss: {avg_clf_loss:.4f}"
    )


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def main(args, create_mb_tiny_fd, create_Mb_Tiny_RFB_fd, DEVICE):
    timer = Timer()

    logging.info(args)
    if args.net == 'slim':
        create_net = create_mb_tiny_fd
        config = fd_config
    elif args.net == 'RFB':
        create_net = create_Mb_Tiny_RFB_fd
        config = fd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, args.overlap_threshold)

    test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    logging.info("Prepare training datasets.")
    if args.dataset_type == 'voc':
        datasets = []
        for dataset_path in args.datasets:
            dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        datasets.append(dataset) 
        train_dataset = ConcatDataset(datasets)
    elif args.dataset_type == 'diy':
        dataset = MyDataset(args.data_base, args.data_split, transform=train_transform, target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        train_dataset = dataset
    else:
        raise ValueError(f"Dataset tpye {args.dataset_type} is not supported.")
    
    logging.info(f"Stored labels into file {label_file}.")
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(
            train_dataset, args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
    )
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(
            args.validation_dataset, 
            transform=test_transform,
            target_transform=target_transform, 
            is_test=True
        )
    elif args.dataset_type == "diy":
        val_dataset = MyDataset(
            args.data_base, 
            args.data_split, 
            transform=test_transform, 
            target_transform=target_transform, 
            is_test=True
        )
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    
    if torch.cuda.device_count() > 1:
        cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        net = nn.DataParallel(net, device_ids=cuda_index_list)
        logging.info("use gpu :{}".format(cuda_index_list))

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=DEVICE)
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
        logging.info("use Adam optimizer")
    else:
        logging.fatal(f"Unsupported optimizer: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, Extra Layers learning rate: {extra_layers_lr}.")
    if args.optimizer_type != "Adam":
        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,
                                    gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
        elif args.scheduler == 'poly':
            logging.info("Uses PolyLR scheduler.")
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.optimizer_type != "Adam":
            if args.scheduler != "poly":
                if epoch != 0:
                    scheduler.step()
        train(train_loader, net, criterion, optimizer, device=DEVICE, epoch=epoch)
        if args.scheduler == "poly":
            adjust_learning_rate(optimizer, epoch)
        # logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            # logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}_{args.input_size}.pth")
            net.save(model_path)
    logging.info(f"Saved model {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train With Pytorch')

    parser.add_argument("--dataset_type", default="diy", type=str, help='Specify dataset type. Currently support voc.')

    parser.add_argument('--data_split', help='Dataset directory path')
    parser.add_argument('--data_base', help='Dataset directory path')
    parser.add_argument('--balance_data', action='store_true', help="Balance training data by down-sampling more frequent labels.")

    parser.add_argument('--net', default="RFB", help="The network architecture ,optional(RFB , slim)")
    parser.add_argument('--freeze_base_net', action='store_true', help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true', help="Freeze all the layers except the prediction head.")
    parser.add_argument('--cuda_index', type=str, help="cuda index for multiple gpu training")

    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-10, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float, help='initial learning rate for base net.')
    parser.add_argument('--extra_layers_lr', default=None, type=float, help='initial learning rate for the layers not in base net and prediction heads.')

    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net', help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str, help="Scheduler for SGD. It can one of multi-step and cosine")

    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', default="80,100", type=str, help="milestones for MultiStepLR")

    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float, help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--batch_size', default=24, type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', default=200, type=int, help='the number epochs')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=5, type=int, help='the number epochs')

    parser.add_argument('--checkpoint_folder', default='my_trained/', help='Directory for saving checkpoint models')
    parser.add_argument('--log_dir', default='./models/Ultra-Light(1MB)_&_Fast_Face_Detector/logs', help='lod dir')
    parser.add_argument('--power', default=2, type=int, help='poly lr pow')
    parser.add_argument('--overlap_threshold', default=0.35, type=float, help='overlap_threshold')
    parser.add_argument('--optimizer_type', default="SGD", type=str, help='optimizer_type')
    parser.add_argument('--input_size', default=320, type=int, help='define network input size,default optional value 128/160/320/480/640/1280')
    args = parser.parse_args()
    define_img_size(args.input_size)
    from models.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from models.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    input_img_size = args.input_size  # define input size ,default optional(128/160/320/480/640/1280)
    logging.info("inpu size :{}".format(input_img_size))
    
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Use {DEVICE}.")
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    main(args, create_mb_tiny_fd, create_Mb_Tiny_RFB_fd, DEVICE)