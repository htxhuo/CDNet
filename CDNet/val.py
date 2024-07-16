import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('run/VisDrone/train/Baseline/weights/best.pt')
    model.val(data='datasets/VisDrone/data.yaml', 
    #model.val(data='datasets/TinyPerson/data.yaml',
                split='val',
                # iou=0.7,
                batch=1,
                #workers=0,
                #imgsz = 1024,
                save_json=True, # if you need to cal coco metrice
                #project='01-03/val', #/iou
                project='run/VisDrone/val', #/iou
                name='Baseline',
                )
    
# 