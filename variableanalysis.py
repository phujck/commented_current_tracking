import numpy as np
import matplotlib.pyplot as plt
import definition as hams
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
import harmonic as har_spec
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    # print(A.size)
    A = np.array(A)
    k = A.size
    # A = np.pad(A, (0, 4 * k), 'constant')
    minus_one = (-1) ** np.arange(A.size)
    # result = np.fft.fft(minus_one * A)
    result = np.fft.fft(minus_one * A)
    # minus_one = (-1) ** np.arange(result.size)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    # print(result.size)
    return result


def smoothing(A, b=1, c=5, d=0):
    if b == 1:
        b = int(A.size / 50)
    if b % 2 == 0:
        b = b + 1
    j = savgol_filter(A, b, c, deriv=d)
    return j


def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c = sys.a * sys.t * 2 * np.imag(conjugator)
    return c


def plot_spectra(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.semilogy(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-15), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectra_switch(U, w, spec, min_spec, max_harm):
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.plot(w, spec[:, i], label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectra_track(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        print(i)
        print(i % 2)
        if i < 2:
            plt.semilogy(w, spec[:, i], label='%s' % (j))
        else:
            plt.semilogy(w, spec[:, i], linestyle='dashed', label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-min_spec), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()




def FT_count(N):
    if N % 2 == 0:
        return int(1 + N / 2)
    else:
        return int((N + 1) / 2)


# These are Parameters I'm using
# number=2
# nelec = (number, number)
# nx = 4®
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10
params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [5 * 3.375, 3.5 * 3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number = 5
number2 = number
nelec = (number, number)
nx = 2*number
nx2 = nx
ny = 0
t = 0.52
t1 = t
t2 = 0.52
# U = 0* t
# U2 = 1* t
U_track=0*t
U_track2=1*t
delta = 0.06
delta1 = delta
delta2 = 0.06
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 1
scalefactor2 = 1
ascale = 1.0
ascale2 = 1.1

# partition=int(nx/2)
partition=7
"""Settings Us lists for loading"""
U_a=0*t
U_b=10*t
U=[]
for n in range(nx):
    if n < partition:
        U.append(U_a)
    else:
        U.append(U_b)
U=np.array(U)
U_a2=0*t
U_b2=0.5*t
U2=[]
for n in range(nx):
    if n < partition:
        U2.append(U_a2)
    else:
        U2.append(U_b2)
U2=np.array(U2)

"""Loading data"""
prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)

# load files
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)

J_field = np.load('./data/variable/Jfield' + parameternames)
phi_original = np.load('./data/variable/phi' + parameternames)
D_densities=np.load('./data/variable/doublons'+parameternames)
up_densities=np.load('./data/variable/ups'+parameternames)
down_densities=np.load('./data/variable/ups'+parameternames)
sites=np.linspace(1,nx,nx)


parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0)
newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2)
J_field2 = np.load('./data/variable/Jfield' + parameternames2)

D_densities2=np.load('./data/variable/doublons'+parameternames2)
up_densities2=np.load('./data/variable/ups'+parameternames2)
down_densities2=np.load('./data/variable/ups'+parameternames2)


plt.subplot(211)
plt.plot(sites,D_densities[0,:],'-x', label='$\\langle D_j\\rangle$')
plt.plot(sites,up_densities[0,:],'--x', label='$\\langle n_{\\uparrow j} \\rangle$')
plt.plot(sites,down_densities[0,:],'-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
plt.vlines(partition +0.5, 0,up_densities[0,:].max(), linestyle='dashed')
plt.ylabel('expectations, $\\frac{U_b}{t_0}= %.1f$' % (U_b/t))

plt.subplot(212)
plt.plot(sites,D_densities2[0,:],'-x', label='$\\langle D_j\\rangle$')
plt.plot(sites,up_densities2[0,:],'--x', label='$\\langle n_{\\uparrow j} \\rangle$')
plt.plot(sites,down_densities2[0,:],'-.x', label='$\\langle n_{\\downarrow j} \\rangle$')
plt.vlines(partition+ 0.5, 0,up_densities2[0,:].max(), linestyle='dashed')
plt.ylabel('expectations, $\\frac{U_b}{t_0}= %.1f$' % (U_b2/t) )
# plt.plot(sites, D_densities.flatten()-up_densities.flatten()*down_densities.flatten(), label='$\\langle D_j\\rangle-\\langle n_\\uparrow j \\rangle \\langle n_\\downarrow j \\rangle$')
plt.xlabel('site')
plt.legend()

figparameters='-%s-nsites-%s-Ua-%s-U_b-%s-n-%s-partition.pdf' % (nx,U_a,U_b,number,partition)
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/groundstate'+figparameters,bbox_inches='tight')
plt.show()

times = np.linspace(0.0, cycles, len(J_field))
times2 = np.linspace(0.0, cycles, len(J_field2))

plt.plot(times,J_field)
plt.show()


plt.subplot(211)
sitelabels=['$D_%s(t)$' % (j) for j in range(1,nx+1)]
print(sitelabels)
plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f$' % (U_b/t))
for i in range(nx):
    plt.plot(times,D_densities[:,i],label=sitelabels[i])
plt.legend()

plt.subplot(212)
plt.plot(times,D_densities2)
plt.ylabel('$D_j(t), \\frac{U_b}{t_0}= %.1f$' % (U_b2/t))
plt.xlabel('Time [cycles]')
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/doublondensities'+figparameters,bbox_inches='tight')
plt.show()


plt.subplot(211)
sitelabels=['$n_{ \\uparrow %s}(t)$' % (j) for j in range(1,nx+1)]
print(sitelabels)
plt.ylabel('$n_{ \\uparrow j} (t), \\frac{U_b}{t_0}= %.1f$' % (U_b/t))
for i in range(nx):
    plt.plot(times,up_densities[:,i],label=sitelabels[i])
plt.legend()

plt.subplot(212)
plt.plot(times,up_densities2)
plt.ylabel('$n_{ \\uparrow j} (t), \\frac{U_b}{t_0}= %.1f$' % (U_b2/t))
plt.xlabel('Time [cycles]')
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/updensities'+figparameters,bbox_inches='tight')
plt.show()

plt.subplot(211)
sitelabels=['$n_{ \\downarrow %s}(t)$' % (j) for j in range(1,nx+1)]
print(sitelabels)
plt.ylabel('$n_{ \\downarrow j} (t), \\frac{U_b}{t_0}= %.1f$' % (U_b/t))
for i in range(nx):
    plt.plot(times,down_densities[:,i],label=sitelabels[i])
plt.legend()

plt.subplot(212)
plt.plot(times,down_densities2)
plt.ylabel('$n_{ \\downarrow j} (t), \\frac{U_b}{t_0}= %.1f$' % (U_b2/t))
plt.xlabel('Time [cycles]')
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/downdensities'+figparameters,bbox_inches='tight')
plt.show()


plt.subplot(211)
sitelabels=['$D_%s(t)-n_{ \\uparrow %s}n_{ \\downarrow %s}(t)$' % (j,j,j) for j in range(1,nx+1)]
print(sitelabels)
plt.ylabel('$D_j - n_{ \\downarrow j}n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f$' % (U_b/t))
for i in range(nx):
    plt.plot(times,D_densities[:,i]-(up_densities*down_densities)[:,i],label=sitelabels[i])
plt.legend()

plt.subplot(212)
plt.plot(times,D_densities2-(up_densities2*down_densities2))
plt.ylabel('$D_j - n_{ \\downarrow j}n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f$' % (U_b2/t))
plt.xlabel('Time [cycles]')
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/differences'+figparameters,bbox_inches='tight')
plt.show()

plt.subplot(211)
sitelabels=['$n_{ \\uparrow %s}-n_{ \\downarrow %s}(t)$' % (j,j) for j in range(1,nx+1)]
print(sitelabels)
plt.ylabel('$n_{ \\downarrow j}-n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f$' % (U_b/t))
for i in range(nx):
    plt.plot(times,up_densities[:,i]-down_densities[:,i],label=sitelabels[i])
plt.legend()

plt.subplot(212)
plt.plot(times,up_densities2-down_densities2)
plt.ylabel('$n_{ \\downarrow j}-n_{ \\uparrow j}, \\frac{U_b}{t_0}= %.1f$' % (U_b2/t))
plt.xlabel('Time [cycles]')
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/spindiff'+figparameters,bbox_inches='tight')
plt.show()


plt.subplot(311)
plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f$' % (U_b/t))
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
# plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
plt.legend()

plt.subplot(312)
plt.plot(times2, J_field2, label='$\\frac{U_b}{t_0}= %.1f$' % (U_b2/t))
plt.ylabel('$J(t)$')
# plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.legend()

U_a=0*t
U_b=0.5*t
U=[]
for n in range(nx):
    if n < partition:
        U.append(U_a)
    else:
        U.append(U_b)
U=np.array(U)
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)

J_field = np.load('./data/variable/Jfield' + parameternames)

plt.subplot(313)
plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f$' % (U_b/t))
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
# plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)
plt.legend()
plt.xlabel('Time [cycles]')
plt.show()

for values in [10*t, 6*t,2*t,0.5*t]:

    U_a = 0 * t
    U_b = values
    U = []
    for n in range(nx):
        if n < partition:
            U.append(U_a)
        else:
            U.append(U_b)
    U = np.array(U)
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, U, t, number, delta, field, F0)

    J_field = np.load('./data/variable/Jfield' + parameternames)

    plt.subplot(211)
    plt.plot(times, J_field, label='$\\frac{U_b}{t_0}= %.1f$' % (U_b/t))
    # plt.xlabel('Time [cycles]')
    plt.ylabel('$J(t)$')
    plt.xlabel('Time [cycles]')
    plt.legend(loc='upper right')

    plt.subplot(212)
    method = 'welch'
    min_spec = 15
    max_harm = 60
    gabor = 'fL'
    exact = np.gradient(J_field, delta)
    w, spec = har_spec.spectrum_welch(exact, delta1)
    w *= 2. * np.pi / prop.field
    plt.semilogy(w, spec, label='$J(t)$')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    axes.set_ylim([10 ** (-min_spec), spec.max()])
    xlines = [2 * i - 1 for i in range(1, 6)]

for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
    plt.xlabel('Harmonic Order')
    plt.ylabel('HHG spectra')
    # plt.legend(loc='upper right')
plt.savefig('/home/phujck/Dropbox/Hubbard interface/plots/spectra'+figparameters,bbox_inches='tight')
plt.show()




