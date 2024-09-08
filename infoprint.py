import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('yaml/CFFA.yaml')
    # model = YOLO('/home/zhangyang/zy/H/CDNet_local/yaml/HCEDR+CFFA+DAB.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()