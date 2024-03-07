#!/usr/bin/env python
# coding: utf-8
# In[1]:

import pandas as pd
import os

in_path  =_""   #'data/df_'
tp__path =_""   #'csv'
out_path = ""

def count_files(path):
    return sum([len(files) for _, _, files in os.walk(path)])


#for para todas as linhas
s = count_files('data/')
df = pd.DataFrame()
for i in range(0,s):
    path   = in_path +  str(i) + tp_path
    df_ler = pd.read_csv(path)
    df     = pd.concat([df,df_ler]) 

df.to_csv(out_path, index=None)