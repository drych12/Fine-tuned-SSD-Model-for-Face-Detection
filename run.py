"""
This code performs real-time detection using a webcam and displays FPS.
"""
import argparse
import sys
import time
import cv2
import torch
import random
import numpy as np

from models.ssd.config.fd_config import define_img_size

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def detect(args, predictor, class_names):
    if args.on_board:
        from jetcam.csi_camera import CSICamera
        cap = CSICamera(capture_device=0, width=args.width, height=args.height)
    else:
        cap = cv2.VideoCapture(0) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap:
        print("Error: Unable to open video file or camera stream.")
        sys.exit(1)
    
    print("Starting real-time detection. Press 'q' to quit.")
    his_fps = []
    while True:
        t = time.time() 
        if args.on_board:
            frame = cap.read()
        else:
            ret, frame = cap.read()
        if frame is None:
            print("Error: Unable to read frame from camera stream.")
            sys.exit()
        image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

        boxes, labels, probs = predictor.predict(image, args.candidate_size // 2, args.threshold)

        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {boxes.size(0)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        fps = 1 / (time.time() - t)
        his_fps.append(fps)
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow("Real-time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(his_fps) >= 1000000:
            print(f"Mean fps is {sum(his_fps) / len(his_fps)}")
            his_fps = []
    print(f"Mean fps is {sum(his_fps) / len(his_fps)}")
    
    if not args.on_board:
        cap.release()
    cv2.destroyAllWindows()

def main(args):
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
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
        sys.exit()
    net.load(model_path)
    net.float()
    
    detect(args, predictor, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time object detection with FPS display')
    parser.add_argument('--net_type', default="slim", type=str, help='The network architecture, optional: RFB or slim')
    parser.add_argument('--input_size', default=320, type=int, help='Network input size, e.g., 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.6, type=float, help='Score threshold')
    parser.add_argument('--candidate_size', default=1500, type=int, help='NMS candidate size')
    parser.add_argument('--on_board', default=False, action='store_true',help='Run on board')
    parser.add_argument('--width', default=640, help='Width of camera')
    parser.add_argument('--height', default=480, help='Height of camera')
    parser.add_argument('--model_path', default="./checkpoints/cartoon-slim-320.pth", help='Path to the trained model')
    parser.add_argument('--label_path', default="./checkpoints/cartoon-labels.txt", help='Path to the labels')
    args = parser.parse_args()
    define_img_size(args.input_size)
    from models.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from models.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    main(args)
