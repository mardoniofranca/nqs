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
from ffnn import *