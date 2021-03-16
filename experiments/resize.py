# 图片分辨率调整
from PIL import Image
import os.path
import glob
def convertpng(file,outdir,width=128,height=128):
    img=Image.open(file)
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(file)))
path = 'D:/Downloads/Cache/Desktop/Temp/AB/'
for file in glob.glob(path+'*.png'):
    convertpng(file,path)
# convertpng('D:/Downloads/Cache/Desktop/1.JPG','D:/Downloads/Cache/Desktop/Temp/')