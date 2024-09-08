import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yaml/DAB.yaml')  #baseline
    #model.load('s.pt') # not loading pretrain weights
    model.train(data='datasets/VisDrone/data.yaml',
    # model.train(data='datasets/TinyPerson/data.yaml',
                cache=False,   # 吃内存
                #pretrained=False, 
                project='run/VisDrone/train',
                # project='run/TinyPerson/train',
                name='DAB',
                imgsz = 640,
                #imgsz = 1024,
                epochs=200,
                batch=8,
                workers=4,  
                close_mosaic=10,
                optimizer='SGD', # using SGD
                #resume='weights/last.pt', # last.pt path
                )
 