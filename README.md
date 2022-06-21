# PVT + ResNet on Semantic Segmentation

# download of weight files:

pvt_small.pth: https://drive.google.com/file/d/1ds9Rb9wRh9IzGV0CZMM0hnS0QAM_qyIF/view

resnet: https://download.pytorch.org/models/resnet34-b627a593.pth

pvt+resnet: https://drive.google.com/file/d/1VIZUdjQHpTzeNcmeGxHl4IbwWv0ySh4t/view?usp=sharing

# download of dataset:

PASCAL VOC 2012:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

SBD (large version of PASCAL VOC 2012): http://home.bharathh.info/pubs/codes/SBD/download.html

Cityscapes: https://www.cityscapes-dataset.com/downloads/

# Attention:
You should build some folders for saving weight and also add some images in the folder, set your own path of dataset firstly.

# Run:
#python main_experiment.py
You can use the weight file or train from scratch, just delete the comment of line 322.
