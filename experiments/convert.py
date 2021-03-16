from __future__ import print_function #输出格式兼容
import os
from PIL import Image

path = 'D:/Downloads/Cache/Desktop/Baselines/DualGAN-master/test/50000/cityscapes/leftImg8bit/frankfurt/'
names = os.listdir(path)
for name in names:
    img =Image.open(path+name)
    print(img.format, img.size, img.mode)