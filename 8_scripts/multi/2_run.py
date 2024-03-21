#!/usr/bin/env python
# coding: utf-8

### libs
import os; import numpy as np;import math 
import matplotlib.pyplot as plt; import matplotlib
from matplotlib.offsetbox import AnchoredText
import json;import pandas as pd; import warnings
import netket as nk; import netket.nn as nknn
import flax.linen as nn;import jax.numpy as jnp
from ffnn import *

### conf
warnings.filterwarnings('ignore')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
pi = math.pi

### params 
size = 8; n_it = 600; net = 'FFNN'

learning_rate = [0.01,0.02,0.1,0.2,0.5]
stoch_rec     = [0.01,0.02,0.1,0.2,0.5]
variat_st     = [512,1008,2016,4032]

### loop run
for v in variat_st:
    for l in learning_rate:
        for s in stoch_rec: 
            w = 160; limit = 281;
            for i in range(w,limit):
                print(i,s,l,v)
                rad = math.radians(i)
                sin = math.sin(rad)
                cos = math.cos(rad)
                j1  = sin
                j2  = cos
                J     = [j1,j2]
                run(i,rad,J,size,net,n_it,l,s,v)
