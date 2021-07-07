'''
Christian Bunker
M^2QM at UF
June 2021

Template:
Use direct_uhf FCI solver to solve single impurity anderson model (siam)
Then bias leads and do time dependent fci following Ruojing's method

pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr

td fci module:
- have to run thru direct_uhf solver
- I use all spin up formalism: only alpha electrons input, only h1e_a and g1e_aa matter to solver
- benchmarked with dot impurity model from ruojings_td_fci.py
- TimeProp is main driver
- turn on bias in leads, pass hamiltonians, molecule, and scf object to time prop
- outputs current and energy vs time


'''

import molecule_5level
import ruojings_td_fci as td
import plot

import time
import numpy as np
import functools
from pyscf import fci, gto, scf, ao2mo

#################################################
#### functions for creating hamiltonians

def h_leads(V, N):
    '''
    create 1e hamiltonian for leads alone
    V is hopping between leads
    N is number of leads (on each side)
    '''
    
    n_lead_sos = 2*N[0] + 2*N[1]; # 2 spin orbs per lead site
    h = np.zeros((n_lead_sos,n_lead_sos));
    
    # iter over lead sites
    for i in range(2*N[0]-2): # i is spin up orb on left side, i+1 spin down

        h[i,i+2] += -V; # left side
        h[i+2,i] += -V; # h.c.
        
    for i in range(2*N[1]-2):
        
        h[n_lead_sos-1-i,n_lead_sos-1-(i+2)] += -V; # right side
        h[n_lead_sos-1-(i+2),n_lead_sos-1-i] += -V; # h.c.
        
    return h; # end h_leads;


def h_chem(mu,N):
    '''
    create 1e hamiltonian for chem potential of leads
    mu is chemical potential of leads
    N is tuple of number of leads on each side
    '''
    
    n_lead_sos = 2*N[0] + 2*N[1]; # 2 spin orbs per lead site
    h = np.zeros((n_lead_sos,n_lead_sos));
    
    # iter over lead sites
    for i in range(2*N[0]): # i is spin up orb on left side, i+1 spin down

        h[i,i] += mu; # left side
        
    for i in range(1,2*N[1]+1):
        h[n_lead_sos-i,n_lead_sos-i] += mu; # right side
        
    return h; # end h chem
    
    
def h_imp_leads(V,N):
    '''
    create 1e hamiltonian for e's hopping on and off impurity levels
    V is hopping between impurity, leads
    N is number of impurity levels
    '''
    
    h = np.zeros((2+2*N+2,2+2*N+2)); # 2N spin orbs on imp, 1st, last 2 are neighboring lead sites
    Liup,Lidown, Riup, Ridown = 0,1,2+2*N, 2+2*N + 1;
    
    # iter over dot sites
    for i in range(2,2+2*N,2): # i is spin up orb on imp, i+1 is spin down
    
        h[Liup,i] += -V;
        h[i,Liup] += -V; # h.c.
        h[Lidown,i+1] += -V;
        h[i+1,Lidown] += -V;
        h[Riup, i] += -V;
        h[i, Riup] += -V;
        h[Ridown,i+1] += -V;
        h[i+1,Ridown] += -V;
        
    return h; # end h imp leads
    
    
def h_dot_1e(V,N):
    '''
    create 1e part of dot hamiltonian
    dot is simple model of impurity
    V is gate voltage (ie onsite energy of dot sites)
    N is number of dot sites
    '''
    
    h = np.zeros((2*N,2*N));
    
    # on site energy for each dot site
    for i in range (2*N):
    
        h[i,i] = V;
        
    return h; # end h dot 1e
    
def h_dot_2e(U,N):
    '''
    create 2e part of dot hamiltonian
    dot is simple model of impurity
    U is hubbard repulsion
    N is number of dot sites
    '''
    
    h = np.zeros((2*N,2*N,2*N,2*N));
    
    # hubbard repulsion when there are 2 e's on same MO
    for i in range(0,N,2): # i is spin up orb, i+1 is spin down
        h[i,i,i+1,i+1] = U;
        h[i+1,i+1,i,i] = U; # switch electron labels
        
    return h; # end h dot 2e

#######################################################
#### functions for manipulating basic hamiltonians

def start_bias(V, dot_is, h1e):
    '''
    Manipulate a pre stitched h1e by turning on bias on leads

    Args:
    - V is bias voltage
    - dot_is is list of spin orb indices which are part of dot
    '''

    assert(type(dot_is) == type([]) );

    # iter over leads
    for i in range(np.shape(h1e)[0]):

        # ignore dot orbs
        if i < dot_is[0]:
            h1e[i,i] = V/2;
        elif i > dot_is[-1]:
            h1e[i,i] = -V/2;

    print("start bias",dot_is, "\n", h1e)
    return h1e;
    
def stitch_h1e(h_imp, h_imp_leads, h_leads, h_bias, n_leads, verbose = 0):
    '''
    stitch together the various parts of the 1e hamiltonian
    the structure of the final should be block diagonal:
    (Left leads)  V_dl
            V_il  (1e imp ham) V_il
                          V_dl (Right leads)
    '''
    
    # number of spin orbs
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads[0] + 2*n_leads[1];
    n_spin_orbs = (n_lead_sos + n_imp_sos);
    
    # combine pure lead ham states
    assert(np.shape(h_leads) == np.shape(h_bias) );#should both be lead sites only
    h_leads = h_leads + h_bias;
    
    # widened ham has leads on outside, dot sites in middle
    h = np.zeros((n_spin_orbs,n_spin_orbs));
    
    # put pure lead elements on top, bottom block diag
    for i in range(2*n_leads[0]):
        for j in range(2*n_leads[0]):
            
            # the first 2*n leads indices are the left leads
            h[i,j] += h_leads[i,j];
            
    for i in range(2*n_leads[1]):
        for j in range(2*n_leads[1]):
        
            # last 2n_lead indices are right leads
            h[n_spin_orbs-1-i,n_spin_orbs-1-j] += h_leads[n_lead_sos-1-i, n_lead_sos-1-j];
        
    # fill in imp and imp-lead elements in middle
    assert(n_imp_sos+4 == np.shape(h_imp_leads)[0]); # 2 spin orbs to left, right
    for i in range(n_imp_sos + 4):
        for j in range(n_imp_sos + 4):
            
            h[2*n_leads[0] - 2 + i, 2*n_leads[0] - 2 + j] += h_imp_leads[i,j];
            if(i>1 and j>1 and i<n_imp_sos+2 and j< n_imp_sos+2): #skip first two, last two rows, columns
                h[2*n_leads[0] - 2 + i, 2*n_leads[0] - 2 + j] += h_imp[i-2,j-2];
            
    if(verbose > 2):
        print("- h_leads + h_bias:\n",h_leads,"\n- h_imp_leads:\n",h_imp_leads,"\n- h_imp:\n",h_imp);
    return h; # end stitch h1e
    
    
def stitch_h2e(h_imp,n_leads,verbose = 0):
    '''
    Put the 2e impurity hamiltonian in the center of the full leads+imp h2e matrix
    h_imp, 4D array, 2e part of impurity hamiltonian
    '''
    
    n_imp_sos = np.shape(h_imp)[0];
    n_lead_sos = 2*n_leads[0] + 2*n_leads[1];
    i_imp = 2*n_leads[0]; # index where imp orbs start
    n_spin_orbs = n_imp_sos + n_lead_sos
    
    h = np.zeros((n_spin_orbs,n_spin_orbs,n_spin_orbs,n_spin_orbs));
    
    for i1 in range(n_imp_sos):
        for i2 in range(n_imp_sos):
            for i3 in range(n_imp_sos):
                for i4 in range(n_imp_sos):
                    h[i_imp+i1,i_imp+i2,i_imp+i3,i_imp+i4] = h_imp[i1,i2,i3,i4];
                    if(verbose > 2): # check 4D tensor by printing nonzero elems
                        if(h_imp[i1,i2,i3,i4] != 0):
                            print("  h_imp[",i1,i2,i3,i4,"] = ",h_imp[i1,i2,i3,i4]," --> h2e[",i_imp+i1,i_imp+i2,i_imp+i3,i_imp+i4,"]");
                        
    return h; # end stitch h2e


def dot_model(nleads, nsites, norbs, nelecs, physical_params,verbose = 0):
    '''
    Run whole SIAM machinery, with impurity a very simple dot model
    Impurity hamiltonian:
    H_imp = H_dot = -V_g sum_i n_i + U n_{i uparrow} n_{i downarrow}
    where i are impurity sites
    
    Args:
    - nleads, tuple of ints of lead sites on left, right
    - nsites, int, num impurity sites
    - norbs, int, total num spin orbs
    - nelecs, tuple of number es, 0 due to All spin up formalism
    - physical params, tuple of t, thyb, Vbias, mu, Vgate, U
    
    Returns: tuple of
    h1e, 1e part of siam ham
    h2e, 2e part of siam ham
    himp, dot part of siam ham only (tuple of 1e, 2e parts)
    mol, gto.mol object which holds some physical params
    scf inst, holds physics: h1e, h2e, mo coeffs etc
    '''
    
    # unpack inputs
    V_leads, V_imp_leads, V_bias, mu, V_gate, U = physical_params;
    
    if(verbose): # print inputs
        print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- mu = ",mu,"\n- V_gate = ",V_gate, "\n- Hubbard U = ",U);

    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = h_leads(V_leads, nleads); # leads only
    hb = h_chem(mu, nleads);   # can addjust lead chemical potential
    hdl = h_imp_leads(V_imp_leads, nsites); # leads talk to dot
    hd = h_dot_1e(V_gate, nsites); # dot
    h1e = stitch_h1e(hd, hdl, hl, hb, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias
    if(verbose > 2):
        print("\n- Full one electron hamiltonian = \n",h1e);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        print("\n- Nonzero h2e elements = ");
    hd2e = h_dot_2e(U,nsites);
    h2e = stitch_h2e(hd2e, nleads, verbose = verbose);
    himp = hd, hd2e; # dot ham only
    
    #### encode physics of dot model in an SCF obj
    
    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    
    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = h2e # put h2e into scf solver
    scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance
        
    return h1e, h2e, himp, mol, scf_inst;
    
    
def mol_model(nleads, nsites, norbs, nelecs, physical_params,verbose = 0):
    '''
    Run whole SIAM machinery, with impurity Silas' molecule
    returns np arrays: 1e hamiltonian, 2e hamiltonian, molecule obj, and scf object

    Args:
    - nleads, tuple of ints, left lead sites, right lead sites
    - nsites, int, num imp sites
    - norbs, num spin orbs (= 2*(nsites + nleads[0]+nleads[1]))
    - nelecs, tuple of up and down e's, 2nd must always be zero in spin up formalism
    - physical params, tuple of:
        lead hopping, imp hopping, bias voltage, chem potential, tuple of mol params specific to Silas' module (see molecule_5level.py)
    '''

    # checks
    assert norbs == 2*(nsites + nleads[0]+nleads[1]);
    assert nelecs[1] == 0;
    assert nelecs[0] <= norbs;

    # unpack inputs
    V_leads, V_imp_leads, V_bias, mu, mol_params = physical_params;
    D, E, alpha, U = mol_params;

    if(verbose): # print inputs
        try:
            print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- mu = ",mu,"\n- D = ",D,"\n- E = ",E, "\n- alpha = ",alpha, "\n- U = ",U, "\n- E/U = ",E/U,"\n- alpha/D = ",alpha/D,"\n- alpha/(E^2/U) = ",alpha*U/(E*E),"\n- alpha^2/(E^2/U) = ",alpha*alpha**U/(E*E) );
        except: 
            print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias)
    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = h_leads(V_leads, nleads); # leads only
    hb = h_chem(mu, nleads);   # chem potential on leads
    hdl = h_imp_leads(V_imp_leads, nsites); # leads talk to dot
    hd = molecule_5level.h1e(nsites*2,D,E,alpha); # Silas' model
    h1e = stitch_h1e(hd, hdl, hl, hb, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias/chem potential
    if(verbose > 2):
        print("\n- Full one electron hamiltonian = \n",h1e);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        print("\n- Nonzero h2e elements = ");
    hd2e = molecule_5level.h2e(2*nsites, U);
    h2e = stitch_h2e(hd2e, nleads, verbose = verbose);

    #### encode physics of dot model in an SCF obj

    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)

    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = h2e # put h2e into scf solver
    scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance
        
    return h1e, h2e, mol, scf_inst;




#####################################
#### get energies

def direct_FCI(h1e, h2e, norbs, nelecs, nroots = 1, verbose = 0):
    '''
    solve gd state with direct FCI
    '''
    
    cisolver = fci.direct_spin1.FCI();
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("\nDirect FCI energies, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ",E_fci);

    return E_fci, v_fci;


def scf_FCI(mol, scf_inst, nroots = 1, verbose = 0):
    '''
    '''

    # init ci solver with ham from molecule inst
    cisolver = fci.direct_uhf.FCISolver(mol);

    # get unpack from scf inst
    h1e = scf_inst.get_hcore(mol);
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);

    # slater determinant coefficients
    mo_a = scf_inst.mo_coeff[0]
    mo_b = scf_inst.mo_coeff[1]
   
    # since we are in UHF formalism, need to split all hams by alpha, beta
    # but since everything is spin blind, all beta matrices are zeros
    h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
    h2e_aa = ao2mo.incore.general(scf_inst._eri, (mo_a,)*4, compact=False)
    h2e_aa = h2e_aa.reshape(norbs,norbs,norbs,norbs)
    h2e_ab = ao2mo.incore.general(scf_inst._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    h2e_ab = h2e_ab.reshape(norbs,norbs,norbs,norbs)
    h2e_bb = ao2mo.incore.general(scf_inst._eri, (mo_b,)*4, compact=False)
    h2e_bb = h2e_bb.reshape(norbs,norbs,norbs,norbs)
    h1e_tup = (h1e_a, h1e_b)
    h2e_tup = (h2e_aa, h2e_ab, h2e_bb)
    
    # run kernel to get exact energy
    E_fci, v_fci = cisolver.kernel(h1e_tup, h2e_tup, norbs, nelecs, nroots = nroots)
    if(verbose):
        print("\nFCI from UHF, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ", E_fci);

    return E_fci, v_fci;

#####################################
#### wrapper functions, test code

def DotCurrentWrapper():
    '''
    Walks thru all the steps for plotting current thru a SIAM. Impurity is a quantum dot
    - construct the biasless hamiltonian, 1e and 2e parts
    - encode hamiltonians in an scf.UHF inst
    - do FCI on scf.UHF to get exact gd state
    - turn on bias to induce current
    - use ruojing's code to do time propagation
    '''

    # top level inputs
    verbose = 5; # passed along throughout to control printing
    np.set_printoptions(suppress=True); # no sci notatation printing

    # set up the hamiltonian
    n_leads = (3,2); # left leads, right leads
    n_imp_sites = 1 # code not flexible enough to change this
    imp_i = [n_leads[0]*2, n_leads[0]*2+1 ]; # should be list for generality
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    nelecs = (int(norbs/2),0);

    # physical params, should always be floats
    V_leads = 1.0; # hopping
    V_imp_leads = 0.4; # hopping
    V_bias = 0; # wait till later to turn on current
    mu = 0; # 0 chemical potential
    V_gate = -0.5; # gate voltage on dot
    U = 1.0; # hubbard repulsion
    params = V_leads, V_imp_leads, V_bias, mu, V_gate, U;

    # get h1e, h2e, and scf implementation of SIAM with dot as impurity
    h1e, h2e, mol, dotscf = dot_model(n_leads, n_imp_sites, norbs, nelecs, params, verbose = verbose);

    # do fci directly from hams
    direct_FCI(h1e, h2e, norbs, nelecs, verbose = verbose);
    
    # from scf instance, do FCI
    E_fci, v_fci = scf_FCI(mol, dotscf, verbose = verbose);

    # prepare in dynamic state by turning on bias
    V_bias = -0.005;
    h1e = start_bias(V_bias, imp_i,h1e);
    if(verbose > 2):
        print(h1e)

    # from fci gd state, do time propagation
    timestop, deltat = 10.0, 0.01 # time prop params
    timevals, energyvals, currentvals = td.TimeProp(h1e, h2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);

    # renormalize current
    currentvals = currentvals*np.pi/abs(V_bias);
    energyvals = energyvals/energyvals[0];

    # plot current vs time
    plot.GenericPlot(timevals,[currentvals, energyvals],labels=["time (dt = "+str(deltat)+")","Current*$\pi / |V_{bias}|$","td-FCI through dot (ASU)"], handles = ["current", "gd state E/$E_{initial}$"]);
    
    # write results to external file
    # should also write code that plots from external file
    
    return; # end dot current wrapper
    
    
def MolCurrentWrapper():
    '''
    Same as DotCurrentWrapper but impurity is Silas' molecule model
    '''
    
    # top level inputs
    verbose = 5; # passed along throughout to control printing
    np.set_printoptions(suppress=True); # no sci notation printing
    quick_run = False;

    # set up the hamiltonian
    if quick_run: # time optimized inputs
        n_leads = (2,2); # left leads, right leads
        n_imp_sites = 5 
        imp_i = [n_leads[0]*2, n_leads[0]*2+n_imp_sites*2 -1 ]; # first imp spin orb to last, inclusive, -1 bc 1 already included in [0] elem
        norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
        nelecs = (int(norbs/2),0); #half filling
    else:
        n_leads = (2,2); # left leads, right leads
        n_imp_sites = 5 
        imp_i = [n_leads[0]*2, n_leads[0]*2+n_imp_sites*2 -1 ]; # first imp spin orb to last, inclusive, -1 bc 1 already included in [0] elem
        norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
        nelecs = (int(norbs/2),0); #half filling
        #nelecs = (n_leads[0]+n_leads[1] + 8,0); #half filling leads, 8 on imp

    # physical params, should always be floats
    # generic siam params
    V_leads = 1.0; # hopping
    V_imp_leads = 0.4; # hopping
    V_bias = 0; # wait till later to turn on current
    mu = 0; # 0 chem potential
    # molecule specific params
    D = 0.5
    E = 0.0
    alpha = 0.0
    U = 0.0; # hubbard repulsion
    mol_params = (D, E, alpha, U);
    params = (V_leads, V_imp_leads, V_bias, mu, mol_params);
    
    # get h1e, h2e, scf object
    h1e, h2e, molobj, molscf = mol_model( n_leads, n_imp_sites, norbs, nelecs, params,verbose = verbose);
    
    # do fci directly from hams
    if quick_run: # this is just a check and we get nothing out of it
        direct_FCI(h1e, h2e, norbs, nelecs, verbose = verbose);
    
    # from scf instance, do FCI
    if quick_run:
        E_fci, v_fci = scf_FCI(molobj, molscf, nroots = 12, verbose = verbose);
        print(E_fci);
        return;
    else:
        E_fci, v_fci = scf_FCI(molobj, molscf, verbose = verbose);
    
    # prepare in dynamic state by turning on bias
    V_bias = -0.005;
    h1e = start_bias(V_bias, imp_i,h1e);
    if(verbose > 2):
        print(h1e);
        
    # from fci gd state, do time propagation
    if quick_run: # short time interval
        timestop, deltat = 1, 0.1 # time prop params
    else:
        timestop, deltat = 20.0, 0.1;
    timevals, energyvals, currentvals = td.TimeProp(h1e, h2e, v_fci, molobj, molscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);

    # renormalize current
    currentvals = currentvals*np.pi/abs(V_bias);

    # plot current vs time
    if quick_run: # long runs just write data
        plot.GenericPlot(timevals,currentvals,labels=["time","Current*$\pi / V_{bias}$","td-FCI through d orbital impurity"]);
    
    # write results to external file
    folderstring = "dat/"
    if quick_run: folderstring += "quick/"
    fstring = folderstring+ "MolCurrentWrapper000_"+str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1]);
    hstring = time.asctime();
    hstring += "\nSpin blind formalism, bias turned on"
    try:
        hstring += "\nInputs:\n- Num. leads = "+str(n_leads)+"\n- Num. impurity sites = "+str(n_imp_sites)+"\n- nelecs = "+str(nelecs)+"\n- V_leads = "+str(V_leads)+"\n- V_imp_leads = "+str(V_imp_leads)+"\n- V_bias = "+str(V_bias)+"\n- D = "+str(D)+"\n- E = "+str(E)+ "\n- alpha = "+str(alpha) +"\n- U = "+str(U)+ "\n- E/U = "+str(E/U)+"\n- alpha/D = "+str(alpha/D)+"\n- alpha/(E^2/U) = "+str(alpha*U/(E*E))+"\n- alpha^2/(E^2/U) = "+str(alpha*alpha**U/(E*E));
    except:
        hstring += "\nInputs:\n- Num. leads = "+str(n_leads)+"\n- Num. impurity sites = "+str(n_imp_sites)+"\n- nelecs = "+str(nelecs)+"\n- V_leads = "+str(V_leads)+"\n- V_imp_leads = "+str(V_imp_leads)+"\n- V_bias = "+str(V_bias)+"\n- D = "+str(D)+"\n- E = "+str(E)+ "\n- alpha = "+str(alpha) +"\n- U = "+str(U);
    np.savetxt(fstring+"_J.txt", np.array([timevals, currentvals]), header = hstring);\
    np.savetxt(fstring+"_E.txt", np.array([timevals, energyvals]), header = hstring);

    # plot from external file
    if quick_run:
        data = np.loadtxt(fstring);
        plot.GenericPlot(data[0], data[1],labels=["time","Current*$\pi / |V_{bias}$|","td-FCI through d orbital impurity"]);
    
    return; # end mol current wrapper


def DebugMolCurrent():
    '''
    Same as DotCurrentWrapper but impurity is Silas' molecule model
    '''
    
    # top level inputs
    verbose = 5; # passed along throughout to control printing
    np.set_printoptions(suppress=True); # no sci notation printing

    # set up the hamiltonian
    n_leads = (1,1); # left leads, right leads
    n_imp_sites = 5 #### need to make code ok with this
    imp_i = [n_leads[0]*2, n_leads[0]*2+n_imp_sites*2 -1 ]; # first imp spin orb to last, inclusive, -1 bc 1 already included in [0] elem 
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    nelecs = (8,0);
    myroots = 12;

    # physical params, should always be floats
    # generic siam params
    V_leads = 1.0; # turn off lead stuff completely to check
    V_imp_leads = 0.4; 
    V_bias = 0; 
    mu = 0; # --> infty to keep e's off lead
    # molecule specific params
    D = 0.5
    E = 0.1
    alpha = 0.01
    U = 1.0; # hubbard repulsion
    mol_params = (D, E, alpha, U);
    params = (V_leads, V_imp_leads, V_bias, mu, mol_params);
    
    # get h1e, h2e, scf object
    h1e, h2e, molobj, molscf = mol_model( n_leads, n_imp_sites, norbs, nelecs, params,verbose = verbose);
    ##### confirmed that h1e, h2e are outputting correctly
    
    # do fci directly from hams
    #direct_FCI(h1e, h2e, norbs, nelecs, verbose = verbose);
    #### confirmed this agrees with scf fci for std physical inputs
    
    # from scf instance, do FCI
    E_fci, v_fci = scf_FCI(molobj, molscf, nroots = myroots, verbose = verbose);
    E_shift = (nelecs[0] - 2)/2 *U - 18*D  # num paired e's/2 *U
    print(E_fci - E_shift);
    #### confirmed that after shift, with mu >> 1 the correct triplet, singlet energies are recovered
    
    # prepare in dynamic state by turning on bias
    V_bias = -0.005;
    #h1e = start_bias(V_bias, imp_i,h1e);
    if(verbose > 2):
        print(h1e);
        
    # do time prop
    if myroots != 1: v_fci = v_fci[0]; # must pass only gd state
    timestop, deltat = 1.0, 0.01 # time prop params #### confirmed for dt = 0.001, gd state enrgy is constant over time at 0 bias
    timevals, energyvals, currentvals = td.TimeProp(h1e, h2e, v_fci, molobj, molscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);
    print(energyvals - E_shift);
    #### confirmed that initial gd state energy matches shifted gd state of 5 level molecule

    # write results to external file
    folderstring = "dat/"
    fstring = folderstring+ "Debug_"+str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])
    hstring = time.asctime();
    hstring += "\nSpin blind formalism, bias turned off, lead sites decoupled"
    hstring += "\nInputs:\n- Num. leads = "+str(n_leads)+"\n- Num. impurity sites = "+str(n_imp_sites)+"\n- nelecs = "+str(nelecs)+"\n- V_leads = "+str(V_leads)+"\n- V_imp_leads = "+str(V_imp_leads)+"\n- V_bias = "+str(V_bias)+"\n- D = "+str(D)+"\n- E = "+str(E)+ "\n- alpha = "+str(alpha) +"\n- U = "+str(U)+ "\n- E/U = "+str(E/U)+"\n- alpha/D = "+str(alpha/D)+"\n- alpha/(E^2/U) = "+str(alpha*U/(E*E))+"\n- alpha^2/(E^2/U) = "+str(alpha*alpha**U/(E*E));
    np.savetxt(fstring+"_J.txt", np.array([timevals, currentvals]), header = hstring);
    np.savetxt(fstring+"_E.txt", np.array([timevals, energyvals]), header = hstring);
    
    return; # end debug mol current



    
#####################################
#### exec code

if(__name__ == "__main__"):

    #DebugMolCurrent();

    # test machinery on garnet's simple dot model
    #ls dat
    MolCurrentWrapper();


