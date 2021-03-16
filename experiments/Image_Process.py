#coding=utf-8
import os
import os.path
from PIL import Image
import random
# 图片切割
def splitimage(src, rownum, colnum, dstpath):
    img = Image.open(src)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')
        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]
        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1
        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')
# 图片水平镜像
def Jpg(dir_line):
    try:
       im=Image.open(dir_line)
    except IOError as er_info:
       print (er_info)
       exit()
    x=im.size[0]
    y=im.size[1]
    img=im.load()
    c = Image.new("RGB",(x,y))
    for i in range (0,x):
        for j in range (0,y):
            w=x-i-1
            h=y-j-1
            rgb=img[w,j] #镜像翻转
            # rgb=img[w,h] #翻转180度
            # rgv=img[i,h] #上下翻转
        c.putpixel([i,j],rgb)
    c.show()
    c.save("fake.jpg")
# 更改图片尺寸大小
'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''
def ResizeImage(filein, fileout, width, height, type):
    img = Image.open(filein)
    out = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
    out.save(fileout, type)

if __name__ == "__main__":
    print('1--图片cutting  2--图片mirror  3--图片resize')
    fun = input('请输入操作号：')
    if fun == '1':
        src = input('请输入图片文件路径：')
        if os.path.isfile(src):
            dstpath = input('请输入图片输出目录（不输入路径则表示使用源图片所在目录）：')
            if (dstpath == '') or os.path.exists(dstpath):
                row = int(input('请输入切割行数：'))
                col = int(input('请输入切割列数：'))
                if row > 0 and col > 0:
                    splitimage(src, row, col, dstpath)
                else:
                    print('无效的行列切割参数！')
            else:
                print('图片输出目录 %s 不存在！' % dstpath)
        else:
            print('图片文件 %s 不存在！' % src)
    if fun == '2':
        name="fa.jpg"
        Jpg(name)
    if fun == '3':
        filein = r'D:/aachen_000000_000019_gtFine_labelIds.png'
        fileout = r'D:/aachen_000000_000019_gtFine_labelIds_l.png'
        width = 256
        height = 128
        type = 'png'
        ResizeImage(filein, fileout, width, height, type)
# frankfurt_000000_011810_leftImg8bit
# frankfurt_000001_008200_leftImg8bit
# frankfurt_000001_030310_leftImg8bit
# frankfurt_000001_052120_leftImg8bit
# frankfurt_000001_056580_leftImg8bit