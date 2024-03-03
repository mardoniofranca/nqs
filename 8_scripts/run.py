#!/usr/bin/env python
# coding: utf-8


import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import AnchoredText
import json
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import math 
from tupa import *


w     = 0
limit = 1
pi    = math.pi
size  = 14  
net   = 'FFNN'
n_it  = 100

for i in range(w,limit):
    rad = math.radians(w)
    sin = math.sin(rad)
    cos = math.cos(rad)
    j1  = sin
    j2  = cos
    J     = [j1,j2]
    run(i,w,rad,J,size,net,n_it)
    w   = w + 1
  

