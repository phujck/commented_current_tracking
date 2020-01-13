import numpy as np


# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve

# Contains lots of important functions.
import definition as definition
# Sets up the lattice for the system
import hub_lats as hub

# These also contain various important observable calculators
import harmonic as har_spec
import observable as observable

# Not strictly necessary, but nice to have.
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc





"""Number of electrons"""
number=3
# this specifically enforces spin up number being equal to spin down
nelec = (number, number)

"""number of sites"""
nx = 6
ny = 0

"""System Parameters"""
t = 0.52
U = 1*t
field= 32.9
F0=10
a=4
cycles = 10

"""Timestep used"""
delta = 0.05

"""these lists get popuproped with the appropriate expectations"""
neighbour = []
energy=[]
doublon_energy=[]
phi_original = []
# This is just to check the phi reconstruction does what it's supposed to.
phi_reconstruct=[0.,0.]
J_field = []
two_body = []
D=[]

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

h= hub.create_1e_ham(prop,True)
N = int(time/(prop.freq*delta))+1
print(N)


"""Set up the ode. Look at the scipy.integrate_ode page for more info."""
r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
r.set_initial_value(psi_temp, 0).set_f_params(prop,time,h)
delta=delta
while r.successful() and r.t < time/prop.freq:
    oldpsi=psi_temp
    r.integrate(r.t + delta)
    psi_temp = r.y
    newtime = r.t

    definition.progress(N, int(newtime / delta))
    neighbour.append(har_spec.nearest_neighbour_new(prop, h, psi_temp))
    J_field.append(har_spec.J_expectation(prop, h, psi_temp, newtime, time))
    phi_original.append(har_spec.phi(prop,newtime,time))
    two_body.append(har_spec.two_body_old(prop, psi_temp))
    D.append(observable.DHP(prop, psi_temp))




del phi_reconstruct[0:2]
np.save('./data/original/Jfield'+parameternames,J_field)
np.save('./data/original/phi'+parameternames,phi_original)
np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
np.save('./data/original/neighbour'+parameternames,neighbour)
# np.save('./data/original/neighbour_check'+parameternames,neighbour_check)
np.save('./data/original/twobody'+parameternames,two_body)
np.save('./data/original/double'+parameternames,D)



