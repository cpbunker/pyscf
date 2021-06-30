'''
Christian Bunker
M^2QM at UF
June 2021

Template:
Use FCI exact diag to solve single impurity anderson model (siam)


pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr

siam.py

'''

import plot
import siam
import ruojings_td_fci as td

import time
import numpy as np
import matplotlib.pyplot as plt

#################################################
#### get current data

def DotCurrentData(n_leads, nelecs, timestop, deltat, mu, V_gate, verbose = 0):
    '''
    More flexible version of siam.py DotDCurrentWrapper with inputs allowing tuning of nelecs, mu, Vgate

    Walks thru all the steps for plotting current thru a SIAM. Impurity is a quantum dot
    - construct the biasless hamiltonian, 1e and 2e parts
    - encode hamiltonians in an scf.UHF inst
    - do FCI on scf.UHF to get exact gd state
    - turn on bias to induce current
    - use ruojing's code to do time propagation

    Args:
    nleads, tuple of ints of left lead sites, right lead sites
    nelecs, tuple of num up e's, 0 due to ASU formalism
    mu, float, chem potential on lead sites
    Vgate, float, onsite energy on dot
    timestop, float, how long to run for
    deltat, float, time step increment
    quick, optional bool, if true test func on short run
    verbose, level of stdout printing

    Returns:
    none, but outputs t, E, J data to /dat/DotCurrentData/ folder
    '''

    # set up the hamiltonian
    n_imp_sites = 1 # dot
    imp_i = [n_leads[0]*2, n_leads[0]*2 + 2*n_imp_sites - 1 ]; # imp sites, inclusive
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    # nelecs left as tunable

    # physical params, should always be floats
    V_leads = 1.0; # hopping
    V_imp_leads = 0.4; # hopping
    V_bias = 0; # wait till later to turn on current
    # chemical potential left as tunable
    # gate voltage on dot left as tunable
    U = 1.0; # hubbard repulsion
    params = V_leads, V_imp_leads, V_bias, mu, V_gate, U;

    # get h1e, h2e, and scf implementation of SIAM with dot as impurity
    h1e, h2e, mol, dotscf = siam.dot_model(n_leads, n_imp_sites, norbs, nelecs, params, verbose = verbose);
    
    # from scf instance, do FCI
    E_fci, v_fci = siam.scf_FCI(mol, dotscf, verbose = verbose);

    # prepare in dynamic state by turning on bias
    V_bias = -0.005;
    h1e = siam.start_bias(V_bias, imp_i,h1e);
    if(verbose > 2):
        print(h1e)

    # from fci gd state, do time propagation
    timevals, energyvals, currentvals = td.TimeProp(h1e, h2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);

    # renormalize current
    currentvals = currentvals*np.pi/abs(V_bias);

    # plot current vs time
    if False:
        plot.GenericPlot(timevals,currentvals,labels=["time","Current*$\pi / |V_{bias}|$","td-FCI on SIAM (All spin up formalism)"]);
    
    # write results to external file
    folderstring = "dat/DotCurrentData/";
    fstring = folderstring+ str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])+"_e"+str(nelecs[0])+"_mu"+str(mu)+"_Vg"+str(V_gate)
    hstring = time.asctime();
    hstring += "\nSpin blind formalism, bias turned off, lead sites decoupled"
    hstring += "\nInputs:\n- Num. leads = "+str(n_leads)+"\n- Num. impurity sites = "+str(n_imp_sites)+"\n- nelecs = "+str(nelecs)+"\n-vi V_leads = "+str(V_leads)+"\n- V_imp_leads = "+str(V_imp_leads)+"\n- V_bias = "+str(V_bias)+"\n- mu = "+str(mu) +"\n- V_gate = "+str(V_gate);
    np.savetxt(fstring+"_J.txt", np.array([timevals, currentvals]), header = hstring);
    np.savetxt(fstring+"_E.txt", np.array([timevals, energyvals]), header = hstring);
    print("Saved t, E, J data to "+fstring);
    
    return; # end dot current wrapper


#################################################
#### manipulate current data

def Fourier(signal, samplerate, angular = False, dominant = 0, shorten = False):
    '''
    Uses the discrete fourier transform to find the frequency composition of the signal

    Args:
    - signal, 1d np array of info vs time
    - samplerate, num data pts per second. Necessary for freq to make any sense

    Returns: tuple of
    1d array of |FT|^2, 1d array of freqs
    '''

    # get vals
    nx = len(signal);
    dx = 1/samplerate;

    # perform fourier transform
    FT = np.fft.fft(signal)
    nu = np.fft.fftfreq(nx, dx); # gets accompanying freqs

    # manipulate data
    FT = FT/nx; # norm missing in np.fft.fft
    FT, nu = np.fft.fftshift(FT), np.fft.fftshift(nu); # puts zero freq at center
    FT = np.absolute(FT)*np.absolute(FT); # get norm squared
    if np.isrealobj(signal): # real signals have only positive freqs
        # truncate FT, nu to nu > 0
        FT, nu = FT[int(nx/2):], nu[int(nx/2):]

    # if asked, convert to omega
    if angular: nu = nu*2*np.pi;

    # if asked, get and return dominant frequencies
    if dominant:

        # get as many of the highest freqs as asked for
        nu_maxvals = np.zeros(dominant);
        for i in range(dominant):

            # get current largest FT val
            imax = np.argmax(FT); # where dominant freq occurs
            nu_maxvals[i] = nu[imax]; # place dominant freq in array

            # get current max out of FT
            FT = np.delete(FT, imax);
            nu = np.delete(nu, imax);

        return nu_maxvals; # end here instead

    # if asked, cut off empty high frequencies  

    return  FT, nu;


#################################################
#### wrappers and test code

def MolCurrentPlot():

    # get current data from txt
    nleads = (2,2);
    nimp = 5;
    fname = "dat/MolCurrentWrapper_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1]);
    xJ, yJ = plot.PlotTxt2D(fname+"_J.txt"); #current
    xE, yE = plot.PlotTxt2D(fname+"_E.txt"); # energy
    yE = yE/yE[0] - 1; # normalize energy to 0
    ti, tf = xJ[0], xJ[-1]
    dt = (tf-ti)/len(xJ);

    # control layout of plots
    ax1 = plt.subplot2grid((4, 3), (0, 0))               # energy spectrum, top left
    ax2 = plt.subplot2grid((4, 3), (0, 1), colspan = 2)               # freqs top right
    ax3 = plt.subplot2grid((4, 3), (1, 0), colspan=3, rowspan=2) # J vs t
    ax4 = plt.subplot2grid((4, 3), (3, 0), colspan=3)            # E vs t

    # plot energy spectrum 
    Energies = [0,1,2,3]; # dummies for now
    xElevels, yElevels = plot.ESpectrumPlot(Energies);
    for E in yElevels: # each energy level is sep line
        ax1.plot(xElevels, E);
    ax1.set_ylabel("Energy (a.u.)")
    ax1.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False);

    # plot frequencies
    Fnorm, freq = Fourier(yJ, 1/dt); # gives dominant frequencies # not yet working
    ax2.plot(freq, Fnorm);
    ax2.set_xlabel("$\omega$ (2$\pi$/s)")

    if False: # compare with dominant freq
        wmax = Fourier(yJ, 1/dt, dominant = 3); # gets dominant freq
        ampl_formax = np.amax(yJ)/2; # amplitude when plotting dominant freq
        ax3.plot(xJ, ampl_formax*(np.sin(wmax[0]*xJ)+np.sin(wmax[2]*xJ) ), linestyle = "dashed")

    # plot J vs t on bottom 
    ax3.plot(xJ, yJ);
    ax3.set_title("td-FCI through $d$ orbital, 2 lead sites on each side");
    ax3.set_xlabel("time (dt = 0.01 s)");
    ax3.set_ylabel("Current*$\pi/V_{bias}$");

    # plot E vs t on bottom  
    ax4.plot(xE, yE);
    ax4.set_xlabel("time (dt = 0.01 s)");
    ax4.set_ylabel("$E/E_i$ - 1");
    #ax4.get_yaxis().get_major_formatter()._set_offset(1)

    # config and show
    plt.tight_layout();
    plt.show();

    return; # end mol current plot


def DebugPlot():

    # plot data from txt
    nleads = (1,1);
    nimp = 5;
    fname = "dat/Debug_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_J.txt";
    x, J = plot.PlotTxt2D(fname)

    # compare with fine time step data
    fname_fine = "dat/Debug_fine_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_J.txt";
    xfine, Jfine = plot.PlotTxt2D(fname_fine)

    # plot
    plt.plot(x,J, label = "dt = 0.1 s");
    plt.plot(xfine, Jfine, label = "dt = 0.01 s", linestyle = "dashed");

    # plot data from txt
    fname = "dat/Debug_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_E.txt";
    x, E = plot.PlotTxt2D(fname)
    E = E/E[0] # norm

    # compare with fine time step data
    fname_fine = "dat/Debug_fine_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_E.txt";
    xfine, Efine = plot.PlotTxt2D(fname_fine)
    Efine = Efine/Efine[0] # norm

    # plot
    fig, (ax1, ax2) = plt.subplots(2,1);
    ax1.plot(x,J, label = "dt = 0.1 s");
    ax1.plot(xfine, Jfine, label = "dt = 0.01 s", linestyle = "dashed");
    ax2.plot(x,E, label = "E, dt = 0.1 s");
    ax2.plot(xfine, Efine, label = "E, dt = 0.01 s", linestyle = "dashed");

    # config and show
    labels = ["Time (s)", "Current*$\pi/V_{bias}$", "td-FCI: $d$ orbital, 1 lead site on each side"]
    #ax1.set_xlabel(labels[0]);
    ax1.set_ylabel(labels[1]);
    ax1.set_title(labels[2]);
    ax1.legend();
    ax2.set_ylabel("Normalized energy");
    ax2.set_xlabel("Time (s)");
    #ax2.legend();
    plt.show();


def DotDataVsVgate():
    '''
    Get current data thru DotCurrentData which generates E, J vs time txt files
    Tune gate voltage
    '''

    # system inputs
    nleads = (4,4);
    nelecs = (nleads[0] + nleads[1] + 1,0); # half filling
    tf = 10.0
    dt = 0.01

    # tunable phys params
    mu = 0
    for Vg in np.linspace(-1.0, 1.0, 5):

        # run code
        DotCurrentData(nleads, nelecs, tf, dt, mu, Vg, verbose = 5);

    
#################################################
#### exec code

if __name__ == "__main__":

    if False:
        xi, xf, nx = 0, 10, 1024;
        dx = (xf - xi)/nx
        x = np.linspace(xi, xf, nx)
        y = np.sin(2*np.pi*x)+np.sin(4*2*np.pi*x)+np.sin(8*2*np.pi*x)

        FT, nu = Fourier(y, 1/dx, shorten = True);
        plt.plot(nu, FT);
        plt.show();

    MolCurrentPlot();
