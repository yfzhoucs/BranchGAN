# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:53:12 2018

@author: Consiousflow
"""
import numpy as np
import os
import math
from sklearn import mixture
import matplotlib.pyplot as plt
from pylab import *
from scipy import stats
path_av = "D:/Downloads/Cache/Desktop/Temp/embedding/AV/"
path_bv = "D:/Downloads/Cache/Desktop/Temp/embedding/BV/"
names_av = os.listdir(path_av)
names_bv = os.listdir(path_bv)
i = 1
obs = np.zeros((0,262144))
for name in names_av:
    av = np.load(path_av+name).reshape((1,262144))
    obs = np.append(obs,av,axis=0)
    print(obs.shape)
    i = i+1
for name in names_bv:
   bv = np.load(path_bv+name).reshape((1,262144))
   obs = np.append(obs,bv,axis=0)
   print(obs.shape)
   i = i+1

clf = mixture.GMM(n_components=3)
clf.fit(obs)
p = clf.score(obs)
print(p)

#for i in range(len(p)):
#  p[i] = math.pow(2,p[i])
#print(p)