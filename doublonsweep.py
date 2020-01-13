import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpmath import nsum
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
import harmonic as har_spec
import des_cre as dc
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc





params = {
   'axes.labelsize': 25,
   'legend.fontsize': 20,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [6, 6],
    'text.usetex': True
    }

plt.rcParams.update(params)

darray=[]
cmap = plt.get_cmap('jet_r')
number = 6
nelec = (number, number)
nx = 12
ny = 0
t = 0.52
# t=1.91
# t=1
list=range(0,110,20)
for xx in list:
    color = cmap((float(xx)-7)/45)
    f=0.1*xx
    neighbour = []
    neighbour_check = []
    energy = []
    doublon_energy = []
    phi_original = []
    J_field = []
    phi_reconstruct = [0., 0.]
    boundary_1 = []
    boundary_2 = []
    two_body = []
    two_body_old = []
    error = []
    D = []
    X = []
    singlon = []
    U = t*f
    delta = 0.05
    cycles = 10
    # field= 32.9
    field = 32.9
    F0 = 10
    a = 4

    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (
    nx, cycles, U, t, number, delta, field, F0)

    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
    print('\n')
    print(vars(lat))
    time = cycles
    psi_temp = harmonic.hubbard(lat)[1].astype(complex)
    init = psi_temp
    h = hub.create_1e_ham(lat, True)
    N = int(time / (lat.freq * delta)) + 1
    print(N)
    times = np.linspace(0.0, cycles, N)
    prop = lat
    r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
    r.set_initial_value(psi_temp, 0).set_f_params(lat, time, h)
    branch = 0
    delta = delta
    while r.successful() and r.t < time / lat.freq:
        oldpsi = psi_temp
        r.integrate(r.t + delta)
        psi_temp = r.y
        newtime = r.t
        # add to expectations

        # double occupancy fails for anything other than half filling.
        # D.append(evolve.DHP(prop,psi))
        harmonic.progress(N, int(newtime / delta))
        D.append(observable.DHP(lat, psi_temp))
    darray.append(D)
    if xx ==list[0] or xx==list[-1]:
        plt.plot(times,D,color=color, label='$\\frac{U}{t_0}=$%s' % (f))
    plt.plot(times,D,color=color)
    plt.xlabel('Time [cycles]')
    plt.ylabel('$D(t)$')

plt.legend()
plt.show()

darray=np.array(darray)
np.save('./data/original/doublonarray_12site',darray)


