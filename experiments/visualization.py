# -*- coding: utf-8 -* 文件头注明编码
# BranchGAN中间特征维度256*32*32图片可视化
from PIL import Image
import os
from pylab import *
import numpy as np
im = np.load("D:/frankfurt_000000_000294_gtFine_color.png.npy")
for n in range(1):
  res = np.zeros((1,1,32,32))
  print(n)
  print(im)
  for i in range(32):
    for j in range(32):
      for k in range(256):
        res[0,0,j,i] = res[0,0,j,i] + im[0,k,j,i]
  # print(res.shape)
  res = res.sum(axis=0)
  # print(res)
  res = res.sum(axis=0)
  # print(res)
  img = Image.fromarray(np.uint8(res),mode='L')
  dir_ = './'
  str1 = str(n) + '.png'
  save_dir = os.path.join(dir_,str1)
  img.save(save_dir,'png')

print(img.format, img.size, img.mode)
width,hight=img.size
print (width,hight)
for i in range(width):
  for j in range(hight):
    print(img.getpixel((i,j)), end='')
    print("  ", end='')
  print("")
