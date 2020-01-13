import numpy as np
import scipy.special as scispec
import numpy.ma as ma
import matplotlib.pyplot as plt
import evolve as evolve 
import observable as observable
import definition as harmonic 
import hub_lats as hub
import harmonic as har_spec
from matplotlib import cm as cm
from scipy.integrate import ode
from scipy.interpolate import interp1d


#NOTE: time is inputted and plotted in terms of cycles, but the actual propagation happens in 'normal' time


#input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)
#they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic

def sigmoid(x,k,x0):
    return 1/(1+np.exp(-k*(x-x0)))

neighbour = []
phi_original = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
two_body_old=[]
error=[]
J_field_track=[]
D_track=[]

number=5
nelec = (number, number)
nx = 10
ny = 0
t = 0.52
# t=1.91
# t=1
U =1.5*t
delta = 0.05
cycles = 10
field= 32.9
# field=25
F0=10
a=4
ascale=1
scalefactor=1
parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (nx,cycles,U,t,number,delta,field,F0)
newparameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (nx,cycles,U,t,number,delta,field,F0,ascale)

# parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (4,cycles,U,t,2,delta,field,F0)
# J_field=np.load('./data/original/Jfield'+parameternames)
# D=np.load('./data/original/double'+parameternames)

lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
time=cycles
N_old = int(time/(lat.freq*delta))+1
oldfreq=lat.freq
times = np.linspace(0.0, cycles/lat.freq, N_old)
# times = np.linspace(0.0, cycles, len(D))


# lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=6*t, t=t, F0=F0, a=ascale*a, bc='pbc')
# times = np.linspace(0.0, cycles/lat.freq, len(J_field))
# times = np.linspace(0.0, cycles, len(D))
print('\n')
print(vars(lat))
psi_temp = harmonic.hubbard(lat)[1].astype(complex)
h= hub.create_1e_ham(lat,True)

N= int(cycles/(lat.freq*delta))+1

real_period=cycles/lat.freq
switch_func= np.array([0.25*sigmoid(x,2,real_period/5)-0.25*sigmoid(x,2,real_period- real_period/5) for x in times])
# switch_func= np.array([0.5*sigmoid(x,1,real_period/2)*(1+0.00001*x) for x in times])
J_field=switch_func
J_func = interp1d(times, scalefactor*J_field, fill_value=0, bounds_error=False, kind='cubic')



delta_track=delta
prop=lat
r = ode(evolve.integrate_f_track_J).set_integrator('zvode', method='bdf')
# r = ode(evolve.integrate_f_track_D).set_integrator('zvode', method='bdf')

# set which observable to track
r.set_initial_value(psi_temp, 0).set_f_params(lat,h,J_func)
# r.set_initial_value(psi_temp, 0).set_f_params(lat,h,D_func)

branch = 0
while r.successful() and r.t < time/lat.freq:
    oldpsi=psi_temp
    r.integrate(r.t + delta_track)
    psi_temp = r.y
    newtime = r.t
    # add to expectations

    harmonic.progress(N, int(newtime / delta_track))
    neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
    # two_body.append(har_spec.two_body_old(lat, psi_temp))

    # tracking current
    phi_original.append(evolve.phi_J_track(lat,newtime,J_func,neighbour[-1],psi_temp))

    # tracking D
    # phi_original.append(evolve.phi_D_track(lat,newtime,D_func,two_body[-1],psi_temp))

    J_field_track.append(har_spec.J_expectation_track(lat, h, psi_temp,phi_original[-1]))
    # D_track.append(observable.DHP(lat, psi_temp))

    # diff = (psi_temp - oldpsi) / delta
    # newerror = np.linalg.norm(diff + 1j * psierror)
    # error.append(newerror)
del phi_reconstruct[0:2]
#

# Cheating to try and track D

# delta_track=delta
# prop=lat
# r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
# r.set_initial_value(psi_temp, 0).set_f_params(lat,time,h)
# branch = 0
# delta=delta
# while r.successful() and r.t < 0.5*time/lat.freq:
#     oldpsi=psi_temp
#     r.integrate(r.t + delta)
#     psi_temp = r.y
#     newtime = r.t
#     # add to expectations
#
#     # double occupancy fails for anything other than half filling.
#     # D.append(evolve.DHP(prop,psi))
#     harmonic.progress(N, int(newtime / delta))
#     psierror=evolve.f(lat,evolve.ham1(lat,h,newtime,time),oldpsi)
#     neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
#     J_field_track.append(har_spec.J_expectation(lat, h, psi_temp, newtime, time))
#     phi_original.append(har_spec.phi(lat,newtime,time))
#     two_body.append(har_spec.two_body_old(lat, psi_temp))
#     D_track.append(observable.DHP(lat, psi_temp))
#
#
# r = ode(evolve.integrate_f_track).set_integrator('zvode', method='bdf')
#
# # set which observable to track
# # r.set_initial_value(psi_temp, 0).set_f_params(lat,h,J_func)
# r.set_initial_value(psi_temp, 0.5*time/lat.freq).set_f_params(lat,h,D_func)
#
# branch = 0
# while r.successful() and r.t < time/lat.freq:
#     oldpsi=psi_temp
#     r.integrate(r.t + delta_track)
#     psi_temp = r.y
#     newtime = r.t
#     # add to expectations
#
#     # double occupancy fails for anything other than half filling.
#     # D.append(evolve.DHP(prop,psi))
#     harmonic.progress(N, int(newtime / delta_track))
#     neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
#     two_body.append(har_spec.two_body_old(lat, psi_temp))
#
#     # tracking current
#     # phi_original.append(evolve.phi_J_track(lat,newtime,J_func,neighbour[-1],psi_temp))
#
#     # tracking D
#     phi_original.append(evolve.phi_D_track(lat,newtime,D_func,two_body[-1],psi_temp))
#
#     J_field_track.append(har_spec.J_expectation_track(lat, h, psi_temp,phi_original[-1]))
#     D_track.append(observable.DHP(lat, psi_temp))
#
#     # diff = (psi_temp - oldpsi) / delta
#     # newerror = np.linalg.norm(diff + 1j * psierror)
#     # error.append(newerror)
# del phi_reconstruct[0:2]
#
np.save('./data/switch/switchfunc'+newparameternames,switch_func)
np.save('./data/switch/Jfield'+newparameternames,J_field_track)
np.save('./data/switch/phi'+newparameternames,phi_original)
np.save('./data/switch/neighbour'+newparameternames,neighbour)
np.save('./data/switch/twobody'+newparameternames,two_body)



#plot_observables(lat, delta=0.02, time=5., K=.1)
# spectra(lat, initial=None, delta=delta, time=cycles, method='welch', min_spec=7, max_harm=50, gabor='fL')
