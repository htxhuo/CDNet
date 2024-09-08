import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('CDNet/weights/VisDrone_CDNet.pt')
    model.predict(source='datasets/VisDrone/images/test',
                project='run/VisDrone/detect',
                name='CDNet',
                save=True,
                # visualize=True # visualize model features maps
                )