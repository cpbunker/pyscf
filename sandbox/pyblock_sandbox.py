'''
pyblock_playground.py

Mess around with pyblock3

https://github.com/block-hczhai/pyblock3-preview
'''

import numpy as np
from pyblock3 import fcidump, hamiltonian

######################################################################
#### test fci dump on 2 site hubbard

# top level inputs
# spin blind = put all electrons as up and make p,q... spin orbitals
nelecs = (2,0); # put in all spins as spin up
norbs = 4;      # spin orbs ie 1alpha, 1beta, 2alpha, 2beta -> 0,1,2,3
nroots = 6;     # find all solns
verbose = 1;

# physical inputs
t = 3.0;
U = 100.0;

# init hamiltonian ( h1e and g2e) as np arrays
# doing it this way forces floats which is a good fail safe
h1e = np.zeros((norbs,norbs));
g2e = np.zeros((norbs,norbs,norbs,norbs));

# 1 particle terms: hopping
h1e[0,2] = t;
h1e[2,0] = t;
h1e[1,3] = t;
h1e[3,1] = t;

# 2 particle terms: hubbard
g2e[0,0,1,1] = U;
g2e[1,1,0,0] = U;  # interchange particle labels
g2e[2,2,3,3] = U;
g2e[3,3,2,2] = U;

# store hamiltonian in fcidump
hdump = fcidump.FCIDUMP(n_sites = norbs, n_elec = nelecs, h1e = h1e, g2e = g2e)
if verbose: print("Created fcidump");

# get hamiltonian from fcidump
h = hamiltonian.Hamiltonian(hdump);
h_mpo = h.build_qc_mpo(); # hamiltonian as matrix product operator (DMRG lingo)

# TODO: pyblock3.algebra only imports w/in GitHub/pyblock3 repo clone. fix this
