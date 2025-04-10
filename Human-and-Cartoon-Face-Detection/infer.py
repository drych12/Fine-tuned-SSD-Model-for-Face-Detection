"""
This code performs real-time detection using a webcam and displays FPS.
"""
import argparse
import sys
import time
import cv2
import torch

from models.ssd.config.fd_config import define_img_size

def detect(img_dir, save_dir ,predictor, class_names, candidate_size, threshold):
    frame = cv2.imread(img_dir)
    image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

    boxes, labels, probs = predictor.predict(image, candidate_size // 2, threshold)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(save_dir, frame)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    define_img_size(args.input_size)
    class_names = [name.strip() for name in open(args.label_path).readlines()]
    
    if args.net_type == 'slim':
        model_path = args.model_path
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=device)
    elif args.net_type == 'RFB':
        model_path = args.model_path
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)
    
    detect(args.on_board, predictor, class_names, args.candidate_size, args.threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time object detection with FPS display')
    parser.add_argument('--net_type', default="slim", type=str, help='The network architecture, optional: RFB or slim')
    parser.add_argument('--input_size', default=320, type=int, help='Network input size, e.g., 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.6, type=float, help='Score threshold')
    parser.add_argument('--candidate_size', default=1500, type=int, help='NMS candidate size')
    parser.add_argument('--on_board', default=False, action='store_true',help='Run on board')
    parser.add_argument('--width', default=640, help='Width of camera')
    parser.add_argument('--height', default=480, help='Height of camera')
    parser.add_argument('--model_path', default="./cartoon/slim_320.pth", help='Path to the trained model')
    parser.add_argument('--label_path', default="./cartoon/labels.txt", help='Path to the labels')
    args = parser.parse_args()
    define_img_size(args.input_size)
    from models.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from models.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    main(args)
