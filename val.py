import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('CDNet/weights/VisDrone_CDNet.pt')
    model.val(data='datasets/VisDrone/data.yaml', 
    #model.val(data='datasets/TinyPerson/data.yaml',
                split='val',
                # split='test',
                # iou=0.7,
                batch=1,
                #workers=0,
                #imgsz = 1024,
                save_json=True, # if you need to cal coco metrice
                # project='run/VisDrone/test', 
                project='run/VisDrone/val', 
                name='CDNet',
                )
    
# 