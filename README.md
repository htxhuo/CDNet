# CDNet
Efficient detection of aerial images based on cross-level deformable feature aggregation

# Abstract
Object detection in unmanned aerial vehicle (UAV) aerial images has become a research hotspot in recent years, but the presence of dense small objects and significant scales and shape variations among objects has posed challenges. To mitigate these issues, we propose a cross-level deformable feature aggregation network (CDNet). Specifically, we design a high-resolution characterization enhancement with deep reduction (HCEDR) structure to extract positional detail features while reducing redundant deep interference. In addition, a cross-level fusiform feature aggregation (CFFA) structure is proposed to fuse multi-scale cross-level feature information and dense small object spatial detail information. Furthermore, to address the challenge of object shape variations caused by varying aerial viewpoints, a deformable attention bottleneck (DAB) module is designed to enhance the model's boundary sensitivity for irregularly shaped objects. Finally, we introduce a new bounding box loss function (Inner-WIoU) that, in addition to reducing the harmful gradient contributions from extreme samples, adjusts auxiliary bounding box sizes to achieve better fitting of ground-truth object bounding boxes, improving model performance and generalization capability. Extensive experiments on the public datasets VisDrone2021 and TinyPerson demonstrate that, compared to advanced methods, CDNet achieves superior detection performance. 

# Datasets
The two datasets used in this research can be downloaded from [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [TinyPerson](https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark).
