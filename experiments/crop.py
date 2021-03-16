# cv剪裁图片
import os
import shutil
import cv2
src_path = './'
ext_name = '.png'
for r, dirs, files in os.walk( src_path ):
    for dir in dirs:
        for file in os.listdir( dir ):
            if file.endswith( ext_name ):
                image = cv2.imread(src_path+dir+'/'+file)
                image2 = image[2:130, 2:130]
                cv2.imshow("image", image2) # 显示图片，后面会讲解
                cv2.waitKey(10) #等待按键
                cv2.imwrite(src_path+dir+'/'+file, image2)