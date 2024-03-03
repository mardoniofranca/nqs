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


class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x


def graph(size):
    # Define custom graph
    edge_colors = []
    for i in range(size):
        edge_colors.append([i, (i+1)%size, 1])
        edge_colors.append([i, (i+2)%size, 2])
    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)    
    return g 



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



def operators(g,bond_operator,bond_color):
     
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    
    # Custom Hamiltonian operator
    op = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
    
    return hi,op



def run(i,w,rad,J,size,net,n_it):
    
    PARAM                      = i
    
    g                          = graph(size)
    
    bond_operator, bond_color  = bonds(J)
     
    hi,op                      = operators(g,bond_operator,bond_color)
   
    
    model = FFNN() #Neural Network
    
    # We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max = 2)

    # Construct the variational state
    vs = nk.vqs.MCState(sa, model, n_samples=1008) #use model

    # We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
    opt = nk.optimizer.Sgd(learning_rate=0.01)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.01)

    # We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
    # Note that we also specify the method to learn the parameters of the wave-function: here we choose the efficient
    # Stochastic reconfiguration (Sr), here in an iterative setup
    gs = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr) #use vs
    
    
    # We need to specify the local operators as a matrix acting on a local Hilbert space 
    sf = []
    sites = []
    structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(0, size):
        for j in range(0, size):
            structure_factor += (nk.operator.spin.sigmaz(hi, i)*nk.operator.spin.sigmaz(hi, j))*((-1)**(i-j))/size
            
    # Run the optimization protocol
    param_file ="log/" + str(PARAM) + "_" + str(size) + "_" + str(n_it)
    gs.run(out=param_file, n_iter=n_it, obs={'Structure Factor': structure_factor})
    
    data=json.load(open(param_file + ".log"))
    iters = data['Energy']['iters']
    energy=data['Energy']['Mean']['real']
    sf=data['Structure Factor']['Mean']['real']
    
    E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
    structure_factor_gs = (ket_gs.T.conj()@structure_factor.to_linear_operator()@ket_gs).real[0,0]

    j1 = J[0]
    j2 = J[1]	
    
    print(PARAM,w,j1,j2,structure_factor_gs,E_gs[0],np.mean(sf[-1:]),np.mean(energy[-1:]))
    
    l = [PARAM,w,rad,j1,j2,structure_factor_gs,E_gs[0],np.mean(sf[-1:]),np.mean(energy[-1:])]
    
    v = []
    
    v.append(l)
        
    df   = pd.DataFrame(v, columns=['i', 'w','rad','j1', 'j2', 'factor_e', 'exact_e_0', 'factor_c', 'calc_e_0'])
    
    
    param_file = "data/" + str(PARAM) + "_" + str(size) + "_" + str(n_it) + "_" + str(net) + ".log"
    
    df.to_csv(param_file)
    
    print(df) 
