import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('run/VisDrone/train/Baseline/weights/best.pt')
    model.predict(source='datasets/VisDrone/images/val',
                project='run/VisDrone/detect',
                name='Baseline',
                save=True,
                # visualize=True # visualize model features maps
                )