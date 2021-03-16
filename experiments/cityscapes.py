from __future__ import print_function #输出格式兼容
import os
from PIL import Image

# StarGAN
# des = 'D:/Downloads/Cache/Desktop/Temp/results-70000-png/'

# src1 = 'D:/Downloads/Cache/Desktop/Baselines/DualGAN-master/test/50000/cityscapes/leftImg8bit/frankfurt/'
# src_names_1 = os.listdir(src1)
# for i in range(1,268):
#     print(i)
#     print(src_names_1[i-1])
#     os.rename(des+str(i)+'-images.png',des+'frankfurt/'+src_names_1[i-1])

# src2 = 'D:/Downloads/Cache/Desktop/Baselines/DualGAN-master/test/50000/cityscapes/leftImg8bit/lindau/'
# src_names_2 = os.listdir(src2)
# for i in range(268,327):
#     print(i)
#     print(src_names_2[i-268])
#     os.rename(des+str(i)+'-images.png',des+'lindau/'+src_names_2[i-268])

# src3 = 'D:/Downloads/Cache/Desktop/Baselines/DualGAN-master/test/50000/cityscapes/leftImg8bit/munster/'
# src_names_3 = os.listdir(src3)
# for i in range(327,501):
#     print(i)
#     print(src_names_3[i-327])
#     os.rename(des+str(i)+'-images.png',des+'munster/'+src_names_3[i-327])

# Combogan
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

# def replace(oldstr,newstr,str):
#     return str.replace(oldstr,newstr)

# def find(str):
#     if str.find('frankfurt') != -1:
#         return 'frankfurt'
#     elif str.find('lindau') != -1:
#         return 'lindau'
#     elif str.find('munster') != -1:
#         return 'munster'

# def mkdir(path):
#     folder = os.path.exists(path)
#     if not folder:
#         os.makedirs(path)

# src = 'D:/Downloads/Cache/Desktop/distancegan/images-150/'
# names = os.listdir(src)
# for name in names:
#     os.rename(src + name, 'D:/Downloads/Cache/Desktop/' + find(name) + '/' + replace('gtFine_color_fake_B','leftImg8bit',name))

# des = 'D:/Downloads/Cache/Desktop/BrGan/AB/'
# des_names = os.listdir(des)

# src = 'D:/Downloads/Cache/Desktop/BrGan/BA/'
# src_names = os.listdir(src)
# i = 0

# for name in des_names:
#     print(name)
#     print(src_names[i])
#     os.rename(des+name,src_names[i])
#     i = i + 1
#

des = 'D:/Personal/Research/BranchGAN/Experiments/Temp/mssim/men2women/Gt/AB/'
des_names = os.listdir(des)

src = 'D:/Downloads/Cache/Desktop/Temp/AB/'
src_names = os.listdir(src)

i = 0
for name in src_names:
    print(name)
    print(des_names[i])
    os.rename(src+name,src+des_names[i])
    i = i + 1

# for i in range(201,401):
#     print(i)
#     print(des_names[i-201])
#     os.rename(src+str(i)+'-images.png',src+des_names[i-201])