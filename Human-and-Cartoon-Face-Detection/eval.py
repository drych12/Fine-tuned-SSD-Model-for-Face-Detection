import argparse
import os
import sys
import tqdm

import cv2
import torch
from torch.utils.data import DataLoader

from models.datasets.my_dataset import MyDataset
from models.nn.multibox_loss import MultiboxLoss
from models.ssd.config.fd_config import define_img_size

def eval(loader, net, predictor, device):
    net.eval()
    acc = 0
    total = 0
    for _, data in tqdm.tqdm(enumerate(loader), total=len(loader)):
        image, boxes, labels = data
        image = image.squeeze(0).numpy()
        boxes = boxes
        label = labels.squeeze(0)[0]

        with torch.no_grad():
            boxes_p, labels_p, probs = predictor.predict(image, 50, 0.6)
            if len(labels_p) > 0:
                for label_p in labels_p:
                    if label_p == label:
                        acc += 1
                        break
            total += 1
    return acc / total

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = [name.strip() for name in open(args.label_path).readlines()]
    if args.net_type == 'slim':
        model_path = args.model_path
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=device)
    elif args.net_type == 'rfb':
        model_path = args.model_path
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)
    val_dataset = MyDataset(
        args.data_base, 
        args.data_split, 
        is_test=True
    )
    val_loader = DataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)
    acc = eval(val_loader, net, predictor, device)
    print(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect faces in a folder of images and compute simple accuracy.')
    parser.add_argument('--net_type', default="slim", type=str, help='The network architecture, optional: RFB (higher precision) or slim (faster).')
    parser.add_argument('--input_size', default=320, type=int, help='Define network input size, default optional values: 128/160/320/480/640/1280.')
    parser.add_argument('--threshold', default=0.6, type=float, help='Score threshold for filtering predictions.')
    parser.add_argument('--candidate_size', default=1500, type=int, help='NMS candidate size.')
    parser.add_argument('--path', default="imgs", type=str, help='Images folder path.')
    parser.add_argument('--label_path', default='checkpoints/cartoon-labels.txt')
    parser.add_argument('--model_path', default='checkpoints/cartoon-slim-320.pth')
    parser.add_argument('--data_split', default="cartoon/split", help='Dataset directory path')
    parser.add_argument('--data_base', default="F:/dataset/face/dataset", help='Dataset directory path')
    args = parser.parse_args()
    define_img_size(args.input_size)
    from models.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from models.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    main(args)