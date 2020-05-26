"""This program runs the hubbard model for variable values of the onsite potential U"""

import os
os.environ["OMP_NUM_THREADS"] = "10"
import numpy as np


# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve

# Contains lots of important functions.
import definition_variable as definition
# Sets up the lattice for the system
import hub_lats as hub

# These also contain various important observable calculators
import harmonic as har_spec
import observable as observable

# Not strictly necessary, but nice to have.
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc
from tqdm import tqdm

params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [4 * 3.375, 3 * 3.375],
    'text.usetex': True
}
plt.rcParams.update(params)

"""Number of electrons"""
number=5
# this specifically enforces spin up number being equal to spin down
nelec = (number, number)

"""number of sites"""
nx = number*2
ny = 0

"""System Parameters"""
t = 0.52
U_a=0*t
U_b = 10*t
field= 32.9
F0=10
a=4
cycles = 10.1
"""Set up the list of onsite potentials"""
U=[]
for n in range(nx):
    # if n < int(nx/2):
    if n < int(3):
        U.append(U_a)
    else:
        U.append(U_b)
U=np.array(U)

# U=np.linspace(0,10,nx)

"""Timestep used"""
delta = 0.06

"""these lists get populated with the appropriate expectations"""
neighbour = []
energy=[]
doublon_energy=[]
phi_original = []
# This is just to check the phi reconstruction does what it's supposed to.
phi_reconstruct=[0.,0.]
J_field = []
two_body = []
D=[]
D_densities=[]
up_densities=[]
down_densities=[]
sites=np.linspace(1,nx,nx)
print(sites)


"""used for saving expectations after simulation"""
parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (nx,cycles,U,t,number,delta,field,F0)

"""class that contains all the essential parameters+scales them. V. IMPORTANT"""
prop = definition.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
print('\n')
print(vars(prop))


time=cycles

"""This sets initial wavefunction as the ground state"""
psi_temp = definition.hubbard(prop)[1].astype(complex)
init=psi_temp
# D_densities.append(observable.doublon_densities(prop,psi_temp))
# up_densities.append(observable.spin_up_densities(prop,psi_temp))
# down_densities.append(observable.spin_down_densities(prop,psi_temp))

# D_densities=np.array(D_densities)
# up_densities=np.array(up_densities)
# down_densities=np.array(down_densities)
# print(np.sum(up_densities) + np.sum(down_densities))
# print(D_densities.shape)
# plt.plot(sites,D_densities.flatten(),'-x', label='$\\langle D_j\\rangle$')
# plt.plot(sites,up_densities.flatten(),'--x', label='$\\langle n_\\uparrow j \\rangle$')
# plt.plot(sites,down_densities.flatten(),'-.x', label='$\\langle n_\\downarrow j \\rangle$')
# plt.plot(sites, D_densities.flatten()-up_densities.flatten()*down_densities.flatten(), label='$\\langle D_j\\rangle-\\langle n_\\uparrow j \\rangle \\langle n_\\downarrow j \\rangle$')
# plt.xlabel('site')
# plt.legend()

# figparameters='-%s-nsites-%s-cycles-%s-Ua-%s-U_b-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.pdf' % (nx,cycles,U_a,U_b,t,number,delta,field,F0)
# plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/groundstate'+figparameters,bbox_inches='tight')
# plt.show()
h= hub.create_1e_ham(prop,True)
N = int(time/(prop.freq*delta))+1
print(N)
print('setup done')

# """Set up the ode. Look at the scipy.integrate_ode page for more info."""
r = ode(evolve.integrate_f_variable).set_integrator('zvode', method='bdf')
r.set_initial_value(psi_temp, 0).set_f_params(prop,time,h)
delta=delta
while r.successful() and r.t < time/prop.freq:
    oldpsi=psi_temp
    r.integrate(r.t + delta)
    psi_temp = r.y
    newtime = r.t
    D_densities.append(observable.doublon_densities(prop, psi_temp))
    up_densities.append(observable.spin_up_densities(prop, psi_temp))
    down_densities.append(observable.spin_down_densities(prop, psi_temp))
    definition.progress(N, int(newtime / delta))
    J_field.append(har_spec.J_expectation(prop, h, psi_temp, newtime, time))
    phi_original.append(har_spec.phi(prop,newtime,time))
    two_body.append(har_spec.two_body_old(prop, psi_temp))
    D.append(observable.DHP(prop, psi_temp))




del phi_reconstruct[0:2]
np.save('./data/variable/Jfield'+parameternames,J_field)
np.save('./data/variable/phi'+parameternames,phi_original)
np.save('./data/variable/doublons'+parameternames,D_densities)
np.save('./data/variable/ups'+parameternames,up_densities)
np.save('./data/variable/downs'+parameternames,down_densities)

