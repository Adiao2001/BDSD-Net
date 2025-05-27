import sys
import argparse
import os
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'




from ultralytics import RTDETR

if __name__ == "__main__":
    model = RTDETR("ultralytics/cfg/models/cfg2024/RTDETR-Neck/MVH_Neck_light.yaml")
    model._load_teacher("best.pt")
    
    results = model.train(data='data.yaml',
                        epochs=300,
                        imgsz=608,
                        workers=6,
                        batch=8,
                        device=0,
                        )