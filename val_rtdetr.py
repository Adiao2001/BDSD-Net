import sys
import argparse
import os

# sys.path.append(r'C:\Users\dell\Desktop\ultralyticsPro--RTDETR') # Path

from ultralytics import YOLO
from ultralytics import RTDETR

def main(opt):
    weights = opt.weights

    model = RTDETR(weights)

    # model.info()
    
    results = model.val(data='data.yaml',
                    imgsz=608,
                    batch=1,
                    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= r'runs/detect/train/weights/best.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)