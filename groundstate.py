import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
from scipy.interpolate import interp1d

import harmonic as har_spec
import des_cre as dc
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc

params = {
    'axes.labelsize': 40,
    # 'legend.fontsize': 28,
    'legend.fontsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2*3.375, 2*3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())

energies=[]
th_energies=[]
numbers=[]
# input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a)
# they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic
L=12
for number in range(0,L+1):
    numbers.append(2*number)
    nelec = (number, number)
    nx = L
    ny = 0
    t = 0.52
    # t=1.91
    # t=1
    U = 0 * t
    delta = 0.001
    cycles = 10
    field = 0
    # field=32.9*0.5
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
    # N = int(time / (lat.freq * delta)) + 1
    # print(N)

    energies.append(observable.energy(psi_temp,lat,h,0,cycles))
    print(energies[-1])
    if number % 2==1:
        th_e=-4*lat.t*(1+np.sum([2*np.cos(2*np.pi*k/nx) for k in range(1,int((number-1)/2+1))]))
    else:
        th_e=-4*lat.t*(1+np.sum([2*np.cos(2*np.pi*k/nx) for k in range(1,int((number-2)/2+1))])+np.cos(2*np.pi*((number-2)/2+1)/nx))
    if number==0:
        th_e=0
    th_energies.append(th_e)
interpolated = interp1d(numbers, th_energies, fill_value=0, bounds_error=False, kind='cubic')
ns=np.linspace(0,20,1000)
# plt.plot(ns,interpolated(ns),linestyle='dashed',label='Interpolation',color='black')
plt.plot(numbers,energies,linestyle='none', color='red', marker='+',label='Numerical',markersize='16')
plt.plot(numbers,th_energies,linestyle='none', color='blue', marker='x',label='Analytic',markersize='14')
plt.xlabel('$N$')
plt.ylabel('$E_g$')
xint = range(0, 2*L+1,2)
plt.xticks(xint)
plt.legend()
plt.show()