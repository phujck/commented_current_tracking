import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpmath import *
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
import harmonic as har_spec
import des_cre as dc
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc
import scipy


# def phi(lat, current_time, cycles):
#     if lat.field == 0.:
#         return 0.
#     else:
#         return (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
#             lat.field * current_time)

def phi(lat,current_time,cycles):
    if lat.field==0.:
        return 0.
    else:
        return (1/lat.field)*(np.sin(lat.field*current_time/(2.*cycles))**2.)*np.sin(lat.field*current_time)

number = 5
nelec = (number, number)
nx = 10
ny = 0
t = 1
field = 32.9
F0 = 7.8
# F0=10
a = 4
cycles = 10
gaps = []
list = []
corrs = []
Ulist = []
freqlist=[]

for f in range(1, 201):
    list.append(0.1 * f)
    U = 0.1 * f * t
    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
    Ulist.append(0.1*lat.U)
    freqlist.append(lat.field)
    # lat.U=f
    # gap_sum=nsum(lambda n: ((1+0.25*(n*lat.U)**2)**0.5 -0.5*n*lat.U)*((-1)**n), [1, inf])
    # print(gap_sum)
    # gap=(lat.U-4+8*gap_sum)
    # print(gap)

    int = lambda x: np.log(x + (x ** 2 - 1) ** 0.5) / np.cosh(2 * np.pi * x / lat.U)
    corr_inverse = scipy.integrate.quad(int, 1, np.inf)[0]
    corrs.append(lat.U / (4 * corr_inverse))

    # chem_integrand = lambda x: scipy.special.jv(1, x) / (x * (1 + np.exp(x * lat.U / 2)))
    # # chem_integrand= lambda x: scipy.special.jv(1,x)/x
    # chem = scipy.integrate.quad(chem_integrand, 0, np.inf, limit=240)

    chem_alt=lambda x: (16/lat.U)*np.sqrt(x**2-1)/(np.sinh(2*np.pi*x/lat.U))
    gap=scipy.integrate.quad(chem_alt, 1, np.inf, limit=240)[0]

    # gap = lat.U - 2* (2 - 4 * chem[0])

    gaps.append(gap)
params = {
    'axes.labelsize': 30,
    'legend.fontsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2*3.375, 2*3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(list,gaps)
ax2.plot(list,corrs)
ax2.set_ylim([0,8])
ax1.set_ylabel('$\\Delta$')
ax2.set_ylabel('$\\xi$')
plt.xlabel('$\\frac{U}{t_0}$')
plt.show()
print(lat.a)
print(lat.F0)
print(lat.field)
breakdown = [a / (2*b *lat.F0) for a, b in zip(gaps, corrs)]
# x = [n for n, i in enumerate(breakdown) if i > 1][0]
# print(float(x))


# plt.plot(list,gaps)
# plt.show()
#
# plt.plot(list[10:],corrs[10:])
# plt.show()

plt.plot(list,breakdown)
plt.plot(list,np.ones(len(list)))
plt.show()
#
# def phi_unit(lat,current_time,cycles):
#     if lat.field==0.:
#         return 0.
#     else:
#         return (np.sin(current_time*np.pi/(cycles))**2.)*np.sin(lat.field*current_time)
#         # return (np.sin(current_time*np.pi/(cycles))**2.)

# lat.field=1/(2*np.pi)
def E_unit(lat, current_time, cycles):
    if lat.field == 0.:
        return 0.
    else:
        # return lat.field * (np.sin(current_time/ (2 * cycles)) ** 2.) * np.cos(
        #     lat.field * current_time) + (np.pi / cycles) * (np.sin(2 * current_time * np.pi / (cycles))) * np.sin(
        #     lat.field * current_time)
        return (np.sin(current_time * lat.field / (2 * cycles)) ** 2.) * np.cos(
            lat.field * current_time) + (1/(2*cycles)) * (np.sin(current_time * lat.field / (cycles))) * np.sin(
            lat.field * current_time)


N = 1000
times = np.linspace(0.0, cycles/lat.freq, N)
phi_list = [phi(lat, t, cycles) for t in times]
plt.plot(times,E_unit(lat,times,cycles))
for a in breakdown[0:7]:
    plt.plot(times,np.ones(len(times))*a)
    # plt.plot(times,-np.ones(len(times))*a)

# plt.plot(times, phi_list)
plt.show()
breaktimes = []
for a in breakdown:
    for j in times:
        if abs(E_unit(lat, j, cycles)) > a:
            breaktimes.append(lat.freq*j)
            break
plt.plot(breaktimes, 10*np.array(Ulist[:len(breaktimes)]))
plt.show()

print(breaktimes)

# np.save('./data/original/breaktimes', breaktimes)  # print(Ulist)
# np.save('./data/original/breaktimes', breaktimes)  # print(Ulist)

