import numpy as np
import evolve as evolve
import definition as harmonic
import observable as observable
import hub_lats as hub
import harmonic as har_spec
from scipy.integrate import ode
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# NOTE: time is inputted and plotted in terms of cycles, but the actual propagation happens in 'normal' time


# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic


neighbour = []
phi_original = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
two_body_old = []
error = []
J_field_cutfreq = []
D_cutfreq = []
Jalt=[]

number = 5
nelec = (number, number)
nx = 10
ny = 0
t = 0.52
# t=1.91
# t=1
U = 7 * t
U_track = 0* t
cutoff = 40
delta = 0.05
cycles = 10
field = 32.9
# field=25
F0 = 10
a = 4
ascale = 1
ascale_track = ascale
scalefactor = 1
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
nx, cycles, U, t, number, delta, field, F0)
cutparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale_track, cutoff)
# parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (4,cycles,U,t,2,delta,field,F0)
phi_cut = np.load('./data/cutfreqs/phi' + cutparameternames)
#
# plt.plot(phi_cut)
# plt.show()

# lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
# time=cycles
# N_old = int(time/(lat.freq*delta))+1
# oldfreq=lat.freq
# times = np.linspace(0.0, cycles/lat.freq, len(J_field))
# times = np.linspace(0.0, cycles, len(D))


lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a, bc='pbc')
times = np.linspace(0.0, cycles / lat.freq, len(phi_cut))
time = cycles
# times = np.linspace(0.0, cycles, len(D))
print('\n')
print(vars(lat))
psi_temp = harmonic.hubbard(lat)[1].astype(complex)
h = hub.create_1e_ham(lat, True)

N = int(cycles / (lat.freq * delta)) + 1

phi_func = interp1d(times, phi_cut, fill_value=0, bounds_error=False, kind='cubic')
print(type(phi_func))
# plt.plot(times,phi_func(times))
# plt.show()
# D_func = interp1d(times, np.gradient(D,delta/(lat.freq)), fill_value=0, bounds_error=False, kind='cubic')

# b=D_func(0)
# expec=har_spec.two_body_old(lat, psi_temp)
# print(expec.real)
# print(expec.imag)
# print(np.absolute(expec))
# print(b)
# print(evolve.phi_D_track(lat,3,D_func,expec,psi_temp))


# for k in range(N):
#     harmonic.progress(N, k)
#     psi_old=psi_temp
#     neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
#     two_body.append(har_spec.two_body_old(lat, psi_temp))
#     D_track.append(observable.DHP(lat, psi_temp))
#
#     # tracking current
#     # phi_original.append(evolve.phi_J_track(lat,k*delta,J_func,neighbour[-1],psi_temp))
#
#     # tracking D
#     phi_original.append(evolve.phi_D_track(lat,k*delta,D_func,two_body[-1],psi_temp))
#
#     J_field_track.append(har_spec.J_expectation_track(lat, h, psi_temp,phi_original[-1]))
#
#     # psierror=evolve.f(lat,evolve.ham_J_track(lat, h, k*delta, J_func,neighbour[-1], psi_old),psi_old)
#     # psi_temp = evolve.RK4_J_track(lat, h, delta, k * delta,J_func,neighbour[-1], psi_temp)
#
#     psierror=evolve.f(lat,evolve.ham_D_track(lat, h, k*delta, D_func,two_body[-1], psi_old),psi_old)
#     psi_temp = evolve.RK4_D_track(lat, h, delta, k * delta,D_func,two_body[-1], psi_temp)
#
#     diff = (psi_temp - psi_old) / delta
#     newerror = np.linalg.norm(diff + 1j * psierror)
#     error.append(newerror)
#
# #
#


delta_track = delta
prop = lat
r = ode(evolve.integrate_f_cutfreqs).set_integrator('zvode', method='bdf')
# r = ode(evolve.integrate_f_track_D).set_integrator('zvode', method='bdf')

# set which observable to track
r.set_initial_value(psi_temp, 0).set_f_params(lat, h, phi_func)
# r.set_initial_value(psi_temp, 0).set_f_params(lat,h,D_func)

branch = 0
while r.successful() and r.t < time / lat.freq:
    oldpsi = psi_temp
    r.integrate(r.t + delta_track)
    psi_temp = r.y
    newtime = r.t
    # add to expectations
    phi_original.append(phi_func(newtime))
    harmonic.progress(N, int(newtime / delta_track))
    J_field_cutfreq.append(har_spec.J_expectation_cutfreq(lat, h, psi_temp, phi_original[-1]))
    neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
    newJ=-2*lat.a*lat.t*np.abs(neighbour[-1])*np.sin(phi_original[-1]-np.angle(neighbour[-1]))
    Jalt.append(newJ)
    phi_reconstruct.append(evolve.phi_reconstruct(lat,J_field_cutfreq[-1],neighbour[-1]))
    D_cutfreq.append(observable.DHP(lat, psi_temp))
    # diff = (psi_temp - oldpsi) / delta
    # newerror = np.linalg.norm(diff + 1j * psierror)
    # error.append(newerror)
del phi_reconstruct[0:2]

np.save('./data/cutfreqs/Jfieldalt' + cutparameternames, Jalt)
np.save('./data/cutfreqs/Jfield' + cutparameternames, J_field_cutfreq)
np.save('./data/cutfreqs/phi_recon' + cutparameternames, phi_reconstruct)
np.save('./data/cutfreqs/doublon' + cutparameternames, D_cutfreq)


# plt.plot(phi_original)
# plt.ylabel('$\\Phi$')
# plt.show()
#
# plt.plot(Jalt)
# plt.show()