# -*- coding: utf-8 -* 文件头注明编码
# 批量重命名
from __future__ import print_function #输出格式兼容
import os
def replace(oldstr,newstr,str):
  return str.replace(oldstr,newstr)
def find(str):
  if str.find('frankfurt') != -1:
    return 'frankfurt'
  elif str.find('lindau') != -1:
    return 'lindau'
  elif str.find('munster') != -1:
    return 'munster'
if __name__ == '__main__':
    path = 'D:/Downloads/Cache/Desktop/Temp/facades/gt/AB/'
    names = os.listdir(path)
    for name in names:
      os.rename(os.path.join(path,name),os.path.join(path,replace('_real_B','',name)))