#!/usr/bin/env python
# coding: utf-8

# ### 1. Configurations

# In[11]:


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
import time


# In[4]:


def graph(L):
    # Define custom graph
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])
    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)    
    return g 


# In[5]:


def bonds(J):
    #Sigma^z*Sigma^z interactions
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz))

    #Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),  
        (J[1] * exchange).tolist(),
    ]

    bond_color = [1, 2, 1, 2]
    
    return bond_operator, bond_color


# In[6]:


def operators(g,bond_operator,bond_color):
     
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    
    # Custom Hamiltonian operator
    op = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
    
    return hi,op


# In[23]:


def run(i,w,rad,J,size,net,n_it,alpha):
    
    PARAM                      = i
    
    g                          = graph(size)
    
    bond_operator, bond_color  = bonds(J)
     
    hi,op                      = operators(g,bond_operator,bond_color)
   
        
    ma = nk.models.RBM(alpha=alpha)

    # Build the sampler
    sa = nk.sampler.MetropolisExchange(hilbert=hi,graph=g)

    # Custom Hamiltonian operator
    op = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)

    # Optimizer
    opt = nk.optimizer.Sgd(learning_rate=0.01)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.1)

    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=2000)

    # The ground-state optimization loop
    gs = nk.VMC(
        hamiltonian=op,
        optimizer=opt,
        preconditioner=sr,
        variational_state=vs)

    # We need to specify the local operators as a matrix acting on a local Hilbert space 
    sf = []
    sites = []
    structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(0, size):
        for j in range(0, size):
            structure_factor += (nk.operator.spin.sigmaz(hi, i)*nk.operator.spin.sigmaz(hi, j))*((-1)**(i-j))/size
      
    
    print('### RBM calculation')
    # Run the optimization protocol
    param_file ="log/" + str(PARAM) + "_" + str(size) + "_" + str(n_it) +  "_" + str(net)
    
    gs.run(out=param_file, n_iter=n_it, obs={'Structure Factor': structure_factor})

    data=json.load(open(param_file + ".log"))
    # Extract the relevant information
    iters_RBM = data["Energy"]["iters"]
    energy_RBM = data["Energy"]["Mean"]
    
    
    E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
    structure_factor_gs = (ket_gs.T.conj()@structure_factor.to_linear_operator()@ket_gs).real[0,0]
    
    j1 = J[0]
    j2 = J[1]
    
    l = [PARAM,w,rad,j1,j2,structure_factor_gs,E_gs[0],iters_RBM[-1],energy_RBM[-1]]
    
    v = []
    
    v.append(l)
        
    df   = pd.DataFrame(v, columns=['i', 'w','rad','j1', 'j2', 'factor_e', 'exact_e_0', 'factor_c', 'calc_e_0'])
    
    param_file = "data/rbm/" + str(PARAM) + "_" + str(size) + "_" + str(n_it) + "_" + str(net) + ".csv"
    
    df.to_csv(param_file, index=None)
    
    print(df)
    


w     = 44
limit = 45
pi    = math.pi
size  = 14  
net   = 'RBM'
n_it  = 100
alpha = 2 # RBM ansatz with alpha=10

for i in range(w,limit):
    rad = math.radians(w)
    sin = math.sin(rad)
    cos = math.cos(rad)
    j1  = sin
    j2  = cos
    J     = [j1,j2]
    run(i,w,rad,J,size,net,n_it, alpha)
    w   = w + 1
