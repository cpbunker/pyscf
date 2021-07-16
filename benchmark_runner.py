'''
Christian Bunker
M^2QM at UF
July 2021

Runner file for benchmarking td-FCI
'''

import ruojings_td_fci
import siam_current

import numpy as np
import matplotlib.pyplot as plt

##################################################################################
#### replicate results from ruojing's code with siam_current module (ASU formalism)

verbose = 5;
nleads = (2,1);

#time info
dt = 0.01;
tf = 4.0;

# run tests
ruojings_td_fci.Test(nleads, dt = dt, tf = tf, verbose = verbose);
siam_current.Test(nleads, dt = dt, tf = tf, verbose = verbose);

##################################################################################
#### test finite size effects ?
