# CDNet
Efficient detection of aerial images based on cross-level deformable feature aggregation

# Abstract
Object detection in unmanned aerial vehicle (UAV) aerial images has become a research hotspot in recent years, but the presence of dense small objects and significant scales and shape variations among objects has posed challenges. To mitigate these issues, we propose a cross-level deformable feature aggregation network (CDNet). Specifically, we design a high-resolution characterization enhancement with deep reduction (HCEDR) structure to extract positional detail features while reducing redundant deep interference. In addition, a cross-level fusiform feature aggregation (CFFA) structure is proposed to fuse multi-scale cross-level feature information and dense small object spatial detail information. Furthermore, to address the challenge of object shape variations caused by varying aerial viewpoints, a deformable attention bottleneck (DAB) module is designed to enhance the model's boundary sensitivity for irregularly shaped objects. Finally, we introduce a new bounding box loss function (Inner-WIoU) that, in addition to reducing the harmful gradient contributions from extreme samples, adjusts auxiliary bounding box sizes to achieve better fitting of ground-truth object bounding boxes, improving model performance and generalization capability. Extensive experiments on the public datasets VisDrone2021 and TinyPerson demonstrate that, compared to advanced methods, CDNet achieves superior detection performance. 

# Datasets
The two datasets used in this research can be downloaded from [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [TinyPerson](https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark).

# Result
VisDrone2021:  
                 Class     Box(P          R      mAP50  mAP50-95):  
                   all     0.594      0.472      0.502      0.307  
            pedestrian      0.67        0.5      0.574      0.286  
                people     0.655      0.435       0.49      0.209  
               bicycle     0.379      0.258      0.241      0.113  
                   car     0.789      0.833      0.865      0.633  
                   van      0.61       0.51      0.546      0.394  
                 truck     0.554      0.411      0.444      0.304  
              tricycle     0.541      0.372       0.39      0.222  
       awning-tricycle     0.369      0.233       0.23      0.143  
                   bus     0.719      0.629      0.657      0.481  
                 motor     0.654      0.537      0.588      0.288  
  
TinyPerson：  
                 Class     Box(P          R      mAP50  mAP50-95):  
                   all     0.567      0.432      0.432      0.156  
