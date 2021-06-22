'''
Time dependent fci code and SIAM example
Author: Ruojing Peng
'''
import plot

from pyscf import lib, fci, cc
from pyscf.fci import direct_uhf, direct_nosym, cistring
import numpy as np
einsum = lib.einsum

################################################################
#### util functions

def make_hop(eris, norb, nelec):
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
    if RK == 4:
        r = ci.r+dr1*h*0.5
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
    Compute the occ of the molecular orbital at index i 0<occ,2
    by encoding n_i as an h1e
    
    Generally to compute stuff follow this formula:
    - put operator in h1e, h2e format
    - use these to construct an eris
    - pass eris and density matrices to compute energy
    '''
    
    # now put dot occ operator in h1 form
    occ = np.zeros((norbs,norbs));
    occ[site_i,site_i] = 1;
    
    # have to store this operator as an eris object
    occ_eris = ERIs(occ, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    occ_val = compute_energy(d1,d2, occ_eris);
    occ_val = np.real(occ_val);
    return occ_val;
    
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
    
################################################################
#### kernel

def kernel(mode, eris, ci, tf, dt, RK=4, dot_i = None, t_dot = None, verbose = 0):
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
        assert(dot_i != None);
        assert(t_dot != None);
    
        return kernel_plot(eris, ci, tf, dt, dot_i, t_dot, RK, verbose);

def kernel_plot(eris, ci, tf, dt, dot_i, t_dot, RK, verbose):
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
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12()
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
        Current = compute_current(dot_i, t_dot, (d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, ci.norb);
        if(verbose > 2):
            print("    time: ", i*dt,", E = ",Energy," Dot occ. = ",Occupancy);
          
        # fill arrays with observables
        t_vals[i] = i*dt;
        energy_vals[i] = Energy
        occ_vals[i] = Occupancy;
        current_vals[i] = Current;
        
    return t_vals, energy_vals, occ_vals, current_vals ;
    
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


def Test():
    '''
    sample calculation of SIAM
    Impurity = one level dot
    '''
    
    from functools import reduce
    from pyscf import gto,scf,ao2mo,symm
    import h5py
    
    # top level inputs
    verbose = 2;

    # physical inputs
    ll = 2 # number of left leads
    lr = 1 # number of right leads
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
    if(verbose):
        print("1. Construct hamiltonian")
    h1e = np.zeros((norb,)*2)
    for i in range(norb):
        if i < norb-1:
            dot = (i==idot or i+1==idot)
            h1e[i,i+1] = -td/t if dot else -1.0
        if i > 0:
            dot = (i==idot or i-1==idot)
            h1e[i,i-1] = -td/t if dot else -1.0
    h1e[idot,idot] = Vg/t # input gate voltage
    # g2e needs modification for all up spin
    g2e = np.zeros((norb,)*4)
    g2e[idot,idot,idot,idot] = U
    
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
    mf.get_ovlp = lambda *args:np.eye(norb)
    symmetry = 8; # perm. symmetry of chemists integrals
    mf._eri = ao2mo.restore(8, g2e, norb) # h2e into scf solver
    mf.kernel(dm0=(Pa,Pb))
    # ground state FCI
    mo_a = mf.mo_coeff[0]
    mo_b = mf.mo_coeff[1]
    cisolver = direct_uhf.FCISolver(mol)
    h1e_a = reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = reduce(np.dot, (mo_b.T, h1e, mo_b))
    if(verbose > 2):
        print("- alpha, beta parts of 1e hamiltonian",h1e_a,"\n",h1e_b);
    g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
    g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
    g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
    g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
    g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
    h1e_mo = (h1e_a, h1e_b)
    g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
    eci, fcivec = cisolver.kernel(h1e_mo, g2e_mo, norb, nelec)
    if(verbose):
        print("2. FCI solution");
        print("- gd state energy, zero bias = ", eci);
    #############
        
    #### do time propagation
    if(verbose):
        print("3. Time propagation")
    for i in range(idot): # introduce bias on leads to get td current
        h1e[i,i] = V/2
    for i in range(idot+1,norb):
        h1e[i,i] = -V/2
    tf = 10.0
    dt = 0.01
    eris = ERIs(h1e, g2e, mf.mo_coeff) # diff h1e than in uhf, thus time dependence
    ci = CIObject(fcivec, norb, nelec)
    kernel_mode = "plot"; # tell kernel whether to return density matrices or arrs for plotting
    t, E, occ, J = kernel(kernel_mode, eris, ci, tf, dt, dot_i = idot, t_dot = td, verbose = verbose);
    
    # plot current vs time
    plot.GenericPlot(t,J,labels=["time","Current","td-FCI with SIAM"]);
    return;
    
    
def OldCodeWrapper():
    # sample calculation of SIAM
    from functools import reduce
    from pyscf import gto,scf,ao2mo,symm 
    import h5py

    ll = 4 # number of left leads
    lr = 3 # number of right leads
    t = 1.0 # lead hopping
    td = 0.4 # dot-lead hopping
    U = 1.0 # dot interaction
    Vg = -0.5 # gate voltage
    V = -0.005 # bias
    norb = ll+lr+1 # total number of sites
    idot = ll # dot index

    # ground state Hamiltonian with zero bias
    h1e = np.zeros((norb,)*2)
    for i in range(norb):
        if i < norb-1:
            dot = (i==idot or i+1==idot)
            h1e[i,i+1] = -td/t if dot else -1.0
        if i > 0:
            dot = (i==idot or i-1==idot)
            h1e[i,i-1] = -td/t if dot else -1.0
    h1e[idot,idot] = Vg/t
    # g2e needs modification for all up spin
    g2e = np.zeros((norb,)*4)
    g2e[idot,idot,idot,idot] = U/t
    # mean-field calculation initialized to half filling
    nelec = int(norb/2), int(norb/2)
    Pa = np.zeros(norb)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    Pb = np.zeros(norb)
    Pb[1::2] = 1.0
    Pb = np.diag(Pb)

    mol = gto.M()
    mol.incore_anyway = True
    mol.nelectron = sum(nelec)
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args:h1e
    mf.get_ovlp = lambda *args:np.eye(norb)
    mf._eri = ao2mo.restore(8, g2e, norb)
    mf.kernel(dm0=(Pa,Pb))
    # ground state FCI
    mo_a = mf.mo_coeff[0]
    mo_b = mf.mo_coeff[1]
    ci = direct_uhf.FCISolver(mol)
    h1e_a = reduce(np.dot, (mo_a.T, mf.get_hcore(), mo_a))
    h1e_b = reduce(np.dot, (mo_b.T, mf.get_hcore(), mo_b))
    g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
    g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
    g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
    g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
    g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
    h1e_mo = (h1e_a, h1e_b)
    g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
    eci, fcivec = ci.kernel(h1e_mo, g2e_mo, norb, nelec)

    # dynamical Hamiltonian
    for i in range(idot):
        h1e[i,i] = V/t/2
    for i in range(idot+1,norb):
        h1e[i,i] = -V/t/2
    # dynamical propagation
#    tf = 1.0
    tf = 10.0
    dt = 0.01
#    dt = 0.005
    eris = ERIs(h1e, g2e, mf.mo_coeff)
    ci = CIObject(fcivec, norb, nelec)
    (d1as, d1bs), (d2aas, d2abs, d2bbs) = kernel(eris, ci, tf, dt)
    f = h5py.File('SIAM_5_01.hdf5','w')
#    f = h5py.File('SIAM_01.hdf5','w')
#    f = h5py.File('SIAM_005.hdf5','w')
    f.create_dataset('h1e', data=h1e)
    f.create_dataset('g2e', data=g2e)
    f.create_dataset('moa', data=mf.mo_coeff[0])
    f.create_dataset('mob', data=mf.mo_coeff[1])
    f.create_dataset('d1as', data=d1as)
    f.create_dataset('d1bs', data=d1bs)
    f.create_dataset('d2aas', data=d2aas)
    f.create_dataset('d2abs', data=d2abs)
    f.create_dataset('d2bbs', data=d2bbs)
    f.close()
    

if __name__ == "__main__":

    Test();
