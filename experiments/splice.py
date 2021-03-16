# -*- coding: utf-8 -*-
# 结果图片拼接代码
import os
from PIL import Image
def replace(oldstr,newstr,str):
    return str.replace(oldstr,newstr)
UNIT_SIZE = 128 # 128*128
TARGET_WIDTH = 5 * UNIT_SIZE # 拼接完后的横向长度为5*128
TARGET_HEIGHT = 64

path = "D:/data/label"
images = [] # 先存储所有的图像的名称
for root, dirs, files in os.walk(path):
    for f in files :
        images.append(f)

for i in range(25): # 6个图像为一组
    imagefile = []
    print(images[i])
    imagefile.append(Image.open("D:/data/label/"+images[i]))
    imagefile.append(Image.open("D:/data/cyclegan/"+images[i]))
    imagefile.append(Image.open("D:/data/ours/"+images[i]))
    imagefile.append(Image.open("D:/data/pix2pix/"+replace('.png','.jpg',images[i])))
    imagefile.append(Image.open("D:/data/ground_truth/"+images[i]))
    print(imagefile)
    target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE))
    left = 0
    right = UNIT_SIZE
    for image in imagefile:
        target.paste(image, (left, 0, right, TARGET_HEIGHT))# 将image复制到target的指定位置中
        left += UNIT_SIZE # left是左上角的横坐标，依次递增
        right += UNIT_SIZE # right是右下的横坐标，依次递增
        quality_value = 100 # quality来指定生成图片的质量，范围是0～100
        target.save("D:/results/"+images[i], quality = quality_value)
    imagefile = []