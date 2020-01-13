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


Tracking = False
Track_Branch = False


def plot_spectra(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.semilogy(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10**(-15), spec.max()])
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
        axes.set_ylim([10**(-min_spec), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectrogram(t, w, spec, min_spec=11, max_harm=60):
    w = w[w <= max_harm]
    t, w = np.meshgrid(t, w)
    spec = np.log10(spec[:len(w)])
    specn = ma.masked_where(spec < -min_spec, spec)
    cm.RdYlBu_r.set_bad(color='white', alpha=None)
    plt.pcolormesh(t, w, specn, cmap='RdYlBu_r')
    plt.colorbar()
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Harmonic Order')
    plt.title('Time-Resolved Emission')
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
    'figure.figsize': [2*3.375, 2*3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number = 5
number2 = number
nelec = (number, number)
nx = 10
nx2 = nx
ny = 0
t = 0.52
t1 = t
t2 = 0.52
U = 0 * t
U2 = 0 * t
delta = 0.05
delta2 = 0.05
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 0.999
scalefactor2 = 1
ascale = 1.
ascale2 = 1
Jscale = 1
cutoff = 60
cutoff2 = 10

Tracking = True
CutSpec = False
Switch = False

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
if Switch:
    prop_switch = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')

# factor=prop.factor
delta1 = delta
delta_switch = 0.05

# load files
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)
J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
phi_reconstruct = np.load('./data/original/phirecon' + parameternames)
neighbour = np.load('./data/original/neighbour' + parameternames)
# neighbour_check = np.load('./data/original/neighbour_check' + parameternames)
# energy = np.load('./data/original/energy' + parameternames)
# doublon_energy = np.load('./data/original/doublonenergy' + parameternames)
# doublon_energy_L = np.load('./data/original/doublonenergy2' + parameternames)
# singlon_energy = np.load('./data/original/singlonenergy' + parameternames)

two_body = np.load('./data/original/twobody' + parameternames)
# two_body_old=np.load('./data/original/twobodyold'+parameternames)
D = np.load('./data/original/double' + parameternames)

error = np.load('./data/original/error' + parameternames)

parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0)
newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2)
J_field2 = np.load('./data/original/Jfield' + parameternames2)
two_body2 = np.load('./data/original/twobody' + parameternames2)
neighbour2 = np.load('./data/original/neighbour' + parameternames2)
phi_original2 = np.load('./data/original/phi' + parameternames2)
# energy2 = np.load('./data/original/energy' + parameternames2)
# doublon_energy2 = np.load('./data/original/doublonenergy' + parameternames2)
# doublon_energy_L2 = np.load('./data/original/doublonenergy2' + parameternames2)
# singlon_energy2 = np.load('./data/original/singlonenergy' + parameternames2)
# error2 = np.load('./data/original/error' + parameternames2)
D2 = np.load('./data/original/double' + parameternames2)

if Tracking:
    # parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    # nx, cycles, U, t, number, delta, field, F0)
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale)

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames) * Jscale*ascale / scalefactor
    phi_track = np.load('./data/tracking/phi' + newparameternames)/scalefactor
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
    two_body_track = np.load('./data/tracking/twobody' + newparameternames)
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    D_track = np.load('./data/tracking/double' + newparameternames)

    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles2, U2, t2, number2, delta2, field, F0, ascale2)

    J_field_track2 = np.load('./data/tracking/Jfield' + newparameternames2) / scalefactor2
    phi_track2 = np.load('./data/tracking/phi' + newparameternames2)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track2 = np.load('./data/tracking/neighbour' + newparameternames2)
    two_body_track2 = np.load('./data/tracking/twobody' + newparameternames2)
    t_track2 = np.linspace(0.0, cycles, len(J_field_track2))
    D_track2 = np.load('./data/tracking/double' + newparameternames2)



times = np.linspace(0.0, cycles, len(J_field))
t = np.linspace(0.0, cycles, len(J_field))
t2 = np.linspace(0.0, cycles, len(J_field2))

# D_func = interp1d(t, D_grad, fill_value=0, bounds_error=False)
# # D_grad_track = np.gradient(D_track, delta_track)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(t, J_field, label='$\\frac{U}{t_0}=0$')
ax2.plot(t2, J_field2, label='$\\frac{U}{t_0}=7$')
ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
ax2.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')

plt.xlabel('Time [cycles]')
plt.show()


plt.subplot(211)
plt.plot(t, J_field, label='$\\frac{U}{t_0}=0$')
if Tracking:
    plt.plot(t_track* prop_track.freq / prop.freq, J_field_track, linestyle='dashed',
         label='Tracked Current')
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)

plt.subplot(212)
plt.plot(t2, J_field2, label='\\frac{U}{t_0}=7')
if Tracking:
    plt.plot(t_track2, J_field_track2, linestyle='dashed',
         label='Tracked current')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.show()



plt.plot(t, phi_original.real, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    # plt.plot(t_track, phi_track.real-np.angle(neighbour_track), label='tracked', linestyle='dashed')
    plt.plot(t_track, phi_track.real, label='tracked', linestyle='dashed')
plt.show()

cross_times_up=[]
cross_times_down=[]
plt.plot(t, phi_original, label='$\\Phi(t)$')
for k in range (1,2):
    if k != 0:
        line=k*np.ones(len(t)) * np.pi / 2
        idx_pos = np.argwhere(np.diff(np.sign(phi_original - line))).flatten()
        idx_neg = np.argwhere(np.diff(np.sign(phi_original + line))).flatten()
        idx_up=min(idx_pos[0],idx_neg[0])
        idx_down=max(idx_pos[-1],idx_neg[-1])
        # idx_up=idx_up[0]
        # idx_down=idx_down[-1]
        # plt.plot(t, line, color='red')
        # plt.plot(t[idx],line[idx], 'ro')
        cross_times_up.append(idx_up)
        cross_times_down.append(idx_down)
# cross_times_up=np.concatenate(cross_times).ravel()
# plt.plot(t[cross_times_up],phi_original[cross_times_up],'go')
# plt.plot(t[cross_times_down],phi_original[cross_times_down],'ro')
# for xc in cross_times_up:
#     plt.hlines(phi_original[xc],0,t[xc],color='green', linestyle='dashed')
# for xc in cross_times_down:
#     plt.hlines(phi_original[xc],t[xc],t[-1],color='red', linestyle='dashed')
# cross_times_up=(t[cross_times_up])
# cross_times_down=(t[cross_times_down])
if Tracking:
    plt.plot(t[:J_field_track.size], phi_track, label='$\\Phi_T(t)$', linestyle='--')
plt.plot(t, np.ones(len(t)) * np.pi / 2, color='red')
plt.plot(t, np.ones(len(t)) * -1 * np.pi / 2, color='red')
plt.yticks(np.arange(-1.5*np.pi, 2*np.pi, 0.5*np.pi),[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-1.5*np.pi, 2*np.pi, .5*np.pi)])
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\Phi(t)$')
plt.show()