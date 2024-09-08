import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

    #  需要在loss中启用Inner-WIoU

if __name__ == '__main__':
    model = YOLO('yaml/InnerWIoU.yaml')  #baseline
    #model.load('s.pt') # not loading pretrain weights
    model.train(data='datasets/VisDrone/data.yaml',
    # model.train(data='datasets/TinyPerson/data.yaml',
                cache=False,   # 吃内存
                #pretrained=False, 
                project='run/VisDrone/train',
                # project='run/TinyPerson/train',
                name='InnerWIoU',
                imgsz = 640,
                #imgsz = 1024,
                epochs=200,
                batch=8,
                workers=4,  
                close_mosaic=10,
                optimizer='SGD', # using SGD
                #resume='weights/last.pt', # last.pt path
                )
