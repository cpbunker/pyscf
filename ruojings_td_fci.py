'''
Time dependent fci code and SIAM example
Author: Ruojing Peng
'''
import plot

from pyscf import lib, fci, scf, gto, ao2mo
from pyscf.fci import direct_uhf, direct_nosym, cistring
import numpy as np
import matplotlib.pyplot as plt
einsum = lib.einsum

################################################################
#### util functions

def make_hop(eris, norb, nelec): # this drives updting of ci wf
    h2e = direct_uhf.absorb_h1e(eris.h1e, eris.g2e, norb, nelec,.5)
    def _hop(c):
        return direct_uhf.contract_2e(h2e, c, norb, nelec)
    return _hop

def compute_update(ci, eris, h, RK=4):
    hop = make_hop(eris, ci.norb, ci.nelec)
    dr1 =  hop(ci.i)
    di1 = -hop(ci.r)
    if RK == 1:
        return dr1, di1
    if RK == 4: # given, r, dr1 compute 4th order RK for dr, ie update of real part of ci wf
        r = ci.r+dr1*h*0.5        # and likewise for imag part
        i = ci.i+di1*h*0.5
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr2 =  hop(i)
        di2 = -hop(r)

        r = ci.r+dr2*h*0.5
        i = ci.i+di2*h*0.5
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr3 =  hop(i)
        di3 = -hop(r)

        r = ci.r+dr3*h
        i = ci.i+di3*h
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr4 =  hop(i)
        di4 = -hop(r)

        dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
        di = (di1+2.0*di2+2.0*di3+di4)/6.0
        return dr, di
        
        
################################################################
#### measure observables from density matrices

def compute_energy(d1, d2, eris, time=None):
    '''
    Ruojing's code
    Computes <H> by
        1) getting h1e, h2e from eris object
        2) contracting with density matrix (NB density matrix is updated at each time step)

    I overload this function by passing it eris w/ arb op x stored
    then ruojings code gets <x> for any eris operator x
    '''

    h1e_a, h1e_b = eris.h1e
    g2e_aa, g2e_ab, g2e_bb = eris.g2e
    h1e_a = np.array(h1e_a,dtype=complex)
    h1e_b = np.array(h1e_b,dtype=complex)
    g2e_aa = np.array(g2e_aa,dtype=complex)
    g2e_ab = np.array(g2e_ab,dtype=complex)
    g2e_bb = np.array(g2e_bb,dtype=complex)
    d1a, d1b = d1
    d2aa, d2ab, d2bb = d2
    # to physicts notation
    g2e_aa = g2e_aa.transpose(0,2,1,3)
    g2e_ab = g2e_ab.transpose(0,2,1,3)
    g2e_bb = g2e_bb.transpose(0,2,1,3)
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)
    # antisymmetrize integral
    g2e_aa -= g2e_aa.transpose(1,0,2,3)
    g2e_bb -= g2e_bb.transpose(1,0,2,3)

    e  = einsum('pq,qp',h1e_a,d1a)
    e += einsum('PQ,QP',h1e_b,d1b)
    e += 0.25 * einsum('pqrs,rspq',g2e_aa,d2aa)
    e += 0.25 * einsum('PQRS,RSPQ',g2e_bb,d2bb)
    e +=        einsum('pQrS,rSpQ',g2e_ab,d2ab)
    return e
    
def compute_occ(site_i, d1, d2, mocoeffs, norbs):
    '''
    Compute the occ of the molecular orbital at index i (molecular orb: 0 < occ < 2)
    by encoding n_i as an h1e
    
    Generally to compute stuff follow this formula:
    - put operator in h1e, h2e format
    - use these to construct an eris
    - pass eris and density matrices to compute energy
    '''
    
    # now put dot occ operator in h1 form
    occ = np.zeros((norbs,norbs));
    occ[site_i,site_i] = 1; # spin up orb
    occ[site_i+1, site_i+1] = 1; # spin down orb
    
    # have to store this operator as an eris object
    occ_eris = ERIs(occ, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    occ_val = compute_energy(d1,d2, occ_eris);
    occ_val = np.real(occ_val);
    return occ_val;
    
def compute_Sz(site_i, d1, d2, mocoeffs, norbs):
    '''
    Compute Sz for the impurity
    '''

    # now put dot occ operator in h1 form
    Sz = np.zeros((norbs,norbs));
    Sz[site_i,site_i] = 1/2; # spin up
    Sz[site_i+1, site_i+1] = -1/2; # spin down

    # have to store this operator as an eris object
    Sz_eris = ERIs(Sz, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    Sz_val = compute_energy(d1,d2, Sz_eris);
    Sz_val = np.real(Sz_val);
    return Sz_val;
    
def compute_current(dot_i,t,d1,d2,mocoeffs,norbs):

    # operator
    J = np.zeros((norbs,norbs));
    J[dot_i - 1,dot_i] = -t/2;
    J[dot_i,dot_i - 1] =  t/2;
    J[dot_i + 1,dot_i] =  t/2;
    J[dot_i,dot_i + 1] = -t/2;
    
    # have to store this operator as an eris object
    J_eris = ERIs(J, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    J_val = compute_energy(d1,d2, J_eris);
    J_val = np.imag(J_val);
    return J_val;
    
def compute_current_spinblind(dot_i,t,d1,d2,mocoeffs,norbs):
    '''
    Compute current through impurity, in ASU formalism
    
    Args:
    dot_i, list, first and last spin orb indices that belong to impurity
    t, float, t_hyb, hopping on and off impurity
    d1, d2, density matrices
    mocoeffs, np arays, coeffs of HF molecular orbs
    norbs, int, total num spin orbs in system
    '''

    # current operator (1e only)
    J = np.zeros((norbs,norbs));

    # iter over dot sites to fill current op
    for doti in range(dot_i[0], dot_i[-1]+1, 2): # doti is up, doti+1 is down # +1 because dot_i is incluse but range is exclusive
        J[doti - 2,doti] = -t/2;  # dot up spin to left up spin # left moving is -
        J[doti+1-2,doti+1] = -t/2; # down to down
        J[doti,doti - 2] =  t/2; # left up spin to dot up spin # hc of 2 above # right moving is +
        J[doti+1, doti+1-2] = t/2; # hc
        J[doti + 2,doti] = t/2;  # up spin to right up spin
        J[doti+1+2,doti+1] = t/2; # down to down
        J[doti,doti + 2] =  -t/2; # hc
        J[doti+1, doti+1+2] = -t/2; # hc
    
    # have to store this operator as an eris object
    J_eris = ERIs(J, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    J_val = compute_energy(d1,d2, J_eris); # this func of ruojings gets <x> for whatever operator x is stored in eris arg
    J_val = np.imag(J_val);
    return J_val;

################################################################
#### kernel

def kernel(mode, eris, ci, tf, dt, RK=4, i_dot = None, t_dot = None, spinblind = False, verbose = 0):
    '''
    Wrapper for the different kernel implementations
    Lots of assertions to prevent confusion
    
    Kernel implementations:
    - std, taken from ruojing, outputs density matrices
    - plot, outputs values of obervables vs time for plotting
    
    All kernel funcs drive time propagation
    Args:
    - eris is an instance of ERIs (see below)
    - ci is an instance of CIObject (see below)
    - tf, dt, floats are final time and time step
    - RK is order of runge kutta, default 4th
    - dot_i is site (MO) index of the dot, not needed if not doing plot, hence defaults to None
    - t_dot is hopping strength between dot, leads, likewise defaults to None
    - verbose prints helpful debugging stuff
    '''

    
    # select specific kernel implementation from mode input
    modes = ["std","plot"];
    assert(mode in modes); # make sure mode input is supported
    
    if(mode == "std"):
    
        return kernel_std(eris,ci,tf,dt,RK);
        
    if(mode == "plot"):
    
        # check inputs
        assert(i_dot != None);
        assert(t_dot != None);
    
        return kernel_plot(eris, ci, tf, dt, i_dot, t_dot, RK, spinblind, verbose);

def kernel_plot(eris, ci, tf, dt, i_dot, t_dot, RK, spinblind, verbose):
    '''
    Kernel for getting observables at each time step, for plotting
    Outputs 1d arrays of time, energy, dot occupancy, current
    Access thru calling kernel (see above) with mode=plot
    '''

    N = int(tf/dt+1e-6)
    d1as = []
    d1bs = []
    d2aas = []
    d2abs = []
    d2bbs = []
    
    # return vals
    t_vals = np.zeros(N+1);
    energy_vals = np.zeros(N+1);
    occ_vals = np.zeros(N+1);
    current_vals = np.zeros(N+1);
    
    # time step loop
    for i in range(N+1):
    
        # density matrices
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12() # since ci wf updated w/ time, so are density matrices
        d1as.append(d1a)
        d1bs.append(d1b)
        d2aas.append(d2aa)
        d2abs.append(d2ab)
        d2bbs.append(d2bb)
        
        # time step
        dr, dr_imag = compute_update(ci, eris, dt, RK) # update state (r, an fcivec) at each time step
        r = ci.r + dt*dr
        r_imag = ci.i + dt*dr_imag # imag part of fcivec
        norm = np.linalg.norm(r + 1j*r_imag) # normalize complex vector
        ci.r = r/norm # update cisolver attributes
        ci.i = r_imag/norm
        
        # compute observables
        Energy = np.real(compute_energy((d1a,d1b),(d2aa,d2ab,d2bb),eris));
        Occupancy = compute_occ(2,(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, ci.norb);
        Sz = compute_Sz(2,(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, ci.norb);
        if spinblind: # differt operator for current in spin blind formalism
            Current = compute_current_spinblind(i_dot, t_dot, (d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, ci.norb);
        else:
            Current = compute_current(i_dot, t_dot, (d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, ci.norb);

        if(verbose > 3):
            print("    time: ", i*dt, "Energy = ",Energy, " Occupancy = ", Occupancy, "<Sz> = ", Sz);

        # fill arrays with observables
        t_vals[i] = i*dt;
        energy_vals[i] = Energy
        #occ_vals[i] = Occupancy;
        current_vals[i] = Current;
        
    return t_vals, energy_vals, current_vals ;
    
def kernel_std(eris, ci, tf, dt, RK):
    '''
    Kernel for td calc copied straight from ruojing
    Outputs density matrices in form (1e alpha, 1e beta), (2e aa, 2e ab, 2e bb)
    Access thru calling kernel (see above) with mode=std
    '''
    N = int(tf/dt+1e-6)
    d1as = []
    d1bs = []
    d2aas = []
    d2abs = []
    d2bbs = []
    for i in range(N+1):
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12()
        d1as.append(d1a)
        d1bs.append(d1b)
        d2aas.append(d2aa)
        d2abs.append(d2ab)
        d2bbs.append(d2bb)

        print('time: ', i*dt)
        dr, di = compute_update(ci, eris, dt, RK)
        r = ci.r + dt*dr
        i = ci.i + dt*di
        norm = np.linalg.norm(r + 1j*i)
        ci.r = r/norm
        ci.i = i/norm
    d1as = np.array(d1as,dtype=complex)
    d1bs = np.array(d1bs,dtype=complex)
    d2aas = np.array(d2aas,dtype=complex)
    d2abs = np.array(d2abs,dtype=complex)
    d2bbs = np.array(d2bbs,dtype=complex)
    return (d1as, d1bs), (d2aas, d2abs, d2bbs)

#####################################################################################
#### class definitions

class ERIs():
    def __init__(self, h1e, g2e, mo_coeff):
        ''' SIAM-like model Hamiltonian
            h1e: 1-elec Hamiltonian in site basis 
            g2e: 2-elec Hamiltonian in site basis
                 chemists notation (pr|qs)=<pq|rs>
            mo_coeff: moa, mob 
        '''
        moa, mob = mo_coeff
        
        h1e_a = einsum('uv,up,vq->pq',h1e,moa,moa)
        h1e_b = einsum('uv,up,vq->pq',h1e,mob,mob)
        g2e_aa = einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_aa = einsum('prxy,xq,ys->prqs',g2e_aa,moa,moa)
        g2e_ab = einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_ab = einsum('prxy,xq,ys->prqs',g2e_ab,mob,mob)
        g2e_bb = einsum('uvxy,up,vr->prxy',g2e,mob,mob)
        g2e_bb = einsum('prxy,xq,ys->prqs',g2e_bb,mob,mob)

        self.mo_coeff = mo_coeff
        self.h1e = h1e_a, h1e_b
        self.g2e = g2e_aa, g2e_ab, g2e_bb

class CIObject():
    def __init__(self, fcivec, norb, nelec):
        '''
           fcivec: ground state uhf fcivec
           norb: size of site basis
           nelec: nea, neb
        '''
        self.r = fcivec.copy() # ie r is the state in slater det basis
        self.i = np.zeros_like(fcivec)
        self.norb = norb
        self.nelec = nelec

    def compute_rdm1(self):
        rr = direct_uhf.make_rdm1s(self.r, self.norb, self.nelec) # tuple of 1 particle density matrices for alpha, beta spin. self.r is fcivec
        # dm1_alpha_pq = <a_p alpha ^dagger a_q alpha
        ii = direct_uhf.make_rdm1s(self.i, self.norb, self.nelec)
        ri = direct_uhf.trans_rdm1s(self.r, self.i, self.norb, self.nelec) # tuple of transition density matrices for alpha, beta spin. 1st arg is a bra and 2nd arg is a ket
        d1a = rr[0] + ii[0] + 1j*(ri[0]-ri[0].T)
        d1b = rr[1] + ii[1] + 1j*(ri[1]-ri[1].T)
        return d1a, d1b

    def compute_rdm12(self):
        # 1pdm[q,p] = \langle p^\dagger q\rangle
        # 2pdm[p,r,q,s] = \langle p^\dagger q^\dagger s r\rangle
        rr1, rr2 = direct_uhf.make_rdm12s(self.r, self.norb, self.nelec)
        ii1, ii2 = direct_uhf.make_rdm12s(self.i, self.norb, self.nelec)
        ri1, ri2 = direct_uhf.trans_rdm12s(self.r, self.i, self.norb, self.nelec)
        # make_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2bb)
        # trans_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2ba, d2bb)
        d1a = rr1[0] + ii1[0] + 1j*(ri1[0]-ri1[0].T)
        d1b = rr1[1] + ii1[1] + 1j*(ri1[1]-ri1[1].T)
        d2aa = rr2[0] + ii2[0] + 1j*(ri2[0]-ri2[0].transpose(1,0,3,2))
        d2ab = rr2[1] + ii2[1] + 1j*(ri2[1]-ri2[2].transpose(3,2,1,0))
        d2bb = rr2[2] + ii2[2] + 1j*(ri2[3]-ri2[3].transpose(1,0,3,2))
        # 2pdm[r,p,s,q] = \langle p^\dagger q^\dagger s r\rangle
        d2aa = d2aa.transpose(1,0,3,2) 
        d2ab = d2ab.transpose(1,0,3,2)
        d2bb = d2bb.transpose(1,0,3,2)
        return (d1a, d1b), (d2aa, d2ab, d2bb)


##########################################################################################################
#### time propagation 

def TimeProp(h1e, h2e, fcivec, mol,  scf_inst, time_stop, time_step, i_dot, t_dot, kernel_mode = "std", verbose = 0):
    '''
    Time propagate an FCI gd state
    The physics of the FCI gd state is encoded in an scf instance
    
    Kernel is driver of time prop
    Kernel gets hamiltonian, and ci wf, which is coeffs of slater dets of HF-determined molecular orbs
    Then updates ci wf at each time step, this in turn updates density matrices
    Contract density matrices at each time step to compute obervables (e.g. compute_energy, compute_current functions)
    '''

    # assertion statements to check inputs
    assert( np.shape(h1e)[0] == np.shape(h2e)[0]);
    assert( type(mol) == type(gto.M() ) );
    assert( type(scf_inst) == type(scf.UHF(mol) ) );
    assert(type(i_dot) == type([]) );

    # unpack
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);
    if(verbose):
        print("\nTime Propagation, norbs = ", norbs, ", nelecs = ", nelecs);

    # time propagation kernel requires
    # - ERIS object to encode hamiltonians
    # - CI object to encode ci states
    eris = ERIs(h1e, h2e, scf_inst.mo_coeff);
    ci = CIObject(fcivec, norbs, nelecs);
    
    # kernel does time prop, NB we assume a spin blind formalism
    return kernel(kernel_mode, eris, ci, time_stop, time_step, i_dot = i_dot, t_dot = t_dot, spinblind = True, verbose = verbose);


###########################################################################################################
#### test code and wrapper funcs

def Test(dt = 0.01, tf = 1.0, verbose = 0):
    '''
    sample calculation of SIAM
    Impurity = one level dot
    '''
    
    import functools

    # physical inputs
    ll = 3 # number of left leads
    lr = 2 # number of right leads
    t = 1.0 # lead hopping
    td = 0.4 # dot-lead hopping
    U = 1.0 # dot interaction
    Vg = -0.5 # gate voltage
    V = -0.005 # bias
    norb = ll+lr+1 # total number of sites
    idot = ll # dot index
    if(verbose):
        print("\nInputs:\n- Left, right leads = ",(ll,lr),"\n- nelecs = ", (int(norb/2), int(norb/2)),"\n- Gate voltage = ",Vg,"\n- Bias voltage = ",V,"\n- Lead hopping = ",t,"\n- Dot lead hopping = ",td,"\n- U = ",U);

    #### make hamiltonian matrices, spin free formalism
    # remember impurity is just one level dot
    
    # make ground state Hamiltonian with zero bias
    td = 0.0
    if(verbose):
        print("1. Construct hamiltonian")
    h1e = np.zeros((norb,)*2)
    for i in range(norb):
        if i < norb-1:
            dot = (i==idot or i+1==idot)
            h1e[i,i+1] = -td if dot else -t
        if i > 0:
            dot = (i==idot or i-1==idot)
            h1e[i,i-1] = -td if dot else -t
    h1e[idot,idot] = Vg # input gate voltage
    # g2e needs modification for all up spin
    g2e = np.zeros((norb,)*4)
    g2e[idot,idot,idot,idot] = U

    for i in range(idot): # introduce bias on leads to get td current
        h1e[i,i] = V/2
    for i in range(idot+1,norb):
        h1e[i,i] = -V/2
    
    if(verbose > 2):
        print("- Full one electron hamiltonian:\n", h1e)
        
    # code straight from ruojing, don't understand yet
    nelec = int(norb/2), int(norb/2) # half filling
    Pa = np.zeros(norb)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    Pb = np.zeros(norb)
    Pb[1::2] = 1.0
    Pb = np.diag(Pb)
    # UHF
    mol = gto.M()
    mol.incore_anyway = True
    mol.nelectron = sum(nelec)
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args:h1e # put h1e into scf solver
    mf.get_ovlp = lambda *args:np.eye(norb) # init overlap as identity
    mf._eri = g2e # put h2e into scf solver
    mf.kernel(dm0=(Pa,Pb))
    # ground state FCI
    mo_a = mf.mo_coeff[0]
    mo_b = mf.mo_coeff[1]
    cisolver = direct_uhf.FCISolver(mol)
    h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
    g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
    g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
    g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
    g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
    g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
    h1e_mo = (h1e_a, h1e_b)
    g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
    eci, fcivec = cisolver.kernel(h1e_mo, g2e_mo, norb, nelec)
    mycisolver = fci.direct_spin1.FCI();
    myE, myv = mycisolver.kernel(h1e, g2e, norb, nelec);
    if(verbose):
        print("2. FCI solution");
        print("- gd state energy, zero bias = ", eci);
        print("- direct spin 1 gd state, zero bias = ",myE," (norbs, nelecs = ",norb,nelec,")")
    #############
        
    #### do time propagation
    if(verbose):
        print("3. Time propagation")
    td = 0.4;
    h1e[idot, idot+1] = -td;
    h1e[idot, idot-1] = -td;
    h1e[idot+1, idot] = -td;
    h1e[idot-1, idot] = -td;
        
    eris = ERIs(h1e, g2e, mf.mo_coeff) # diff h1e than in uhf, thus time dependence
    ci = CIObject(fcivec, norb, nelec)
    kernel_mode = "plot"; # tell kernel whether to return density matrices or arrs for plotting
    t, E, J = kernel(kernel_mode, eris, ci, tf, dt, i_dot = idot, t_dot = td, verbose = verbose);

    # normalize vals
    J = J*np.pi/abs(V);
    E = E/E[0] - 1;
    
    # plot current vs time
    #plot.GenericPlot(t,[J, E],labels=["time (dt = "+str(dt)+")","J*$\pi / |V_{bias}|$","Dot impurity, 3 left sites, 2 right sites"], handles = ["J", "E/$E_{i} - 1$"]);
    # plot J, E on different axes
    fig, axes = plt.subplots(2, sharex = True);
    axes[0].plot(t, J);
    axes[0].set_xlabel("time (dt = "+str(dt)+")");
    axes[0].set_ylabel("J*$\pi / |V_{bias}|$");
    axes[0].set_title("Dot impurity, 3 left sites, 2 right sites");
    axes[1].plot(t, E);
    axes[1].set_xlabel("time (dt = "+str(dt)+")");
    axes[1].set_ylabel("$E/E_{i} - 1$");
    plt.show();

    return; # end test
    

if __name__ == "__main__":

    Test();
