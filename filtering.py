import numpy as np
import matplotlib.pyplot as plt
import definition as hams
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
import harmonic as har_spec
import definition as harmonic
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import ode
import scipy


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
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.plot(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
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
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        print(i)
        print(i % 2)
        if i < 2:
            plt.plot(w, spec[:, i], label='%s' % (j))
        else:
            plt.plot(w, spec[:, i], linestyle='dashed', label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
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
# nx = 4
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10
params = {
    'axes.labelsize': 30,
    'legend.fontsize': 19,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.figsize': [6, 6],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number = 5
number2 = 5
nelec = (number, number)
nx = 10
nx2 = 10
ny = 0
t = 0.52
t1 = t
t2 = 0.52
U = 7 * t
U2 = 7 * t
delta = 0.05
delta2 = 0.05
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 1
scalefactor2 = 1
ascale = 1.001
ascale2 = 1
Jscale = 1

cutoff=100
cutoff2=cutoff

Tracking = True
prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=1 * t, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=1* t, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
# factor=prop.factor
delta1 = delta
delta_switch = 0.05

# parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
# nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)

J_field_track = np.load('./data/tracking/Jfield' + newparameternames) * Jscale / scalefactor
phi_track = np.load('./data/tracking/phi' + newparameternames)



newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles2, U2, t2, number2, delta2, field, F0, ascale2)

J_field_track2 = np.load('./data/tracking/Jfield' + newparameternames2) / scalefactor2
phi_track2 = np.load('./data/tracking/phi' + newparameternames2)

# delta2=delta2*factor

times = np.linspace(0.0, cycles, len(J_field_track))
times2 = np.linspace(0.0, cycles, len(J_field_track2))

"""Shifting spectrum"""
phi_track_shift=np.copy(phi_track)
if Tracking:
    for j in range(1, int(len(phi_track_shift))):
        k = phi_track_shift[j] - phi_track_shift[j - 1]
        if k > 1.8 * np.pi:
            phi_track_shift[j:] = phi_track_shift[j:] - 2 * np.pi
        if k < -1.8* np.pi:
            phi_track_shift[j:] = phi_track_shift[j:] + 2 * np.pi
    plt.plot(times, phi_track_shift.real, label='shifted phi')
    plt.plot(times,np.zeros(len(times)))
    print(np.sum(phi_track_shift.real)*delta_track+delta_track*len(phi_track_shift)*2.105*np.pi)
    print(np.sum(phi_track.real)*delta_track)
    # plt.legend()
    plt.xlabel('Time [cycles]')
    plt.ylabel('$\\Phi(t)$')
    plt.yticks(np.arange(-3 * np.pi, 3 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-3 * np.pi, 3 * np.pi, .5 * np.pi)])
    plt.show()


"""Cutting the spectrum in order to evolve it"""
# cutoff=5
# cutoff2=cutoff
phi_f1=FT(phi_track)
phi_f2=FT(phi_track_shift)
# phi_f1=np.fft.fftshift(np.fft.fft(phi_track))
# phi_f2=np.fft.fft(phi_track2)
w_phi1= np.fft.fftshift(np.fft.fftfreq(len(phi_track),delta))
w_phi1*= 2. * np.pi / prop_track.field

w_phi2= np.fft.fftshift(np.fft.fftfreq(len(phi_track2),delta))
w_phi2*= 2. * np.pi / prop_track2.field
# plt.plot(w_phi1, np.log10(np.abs(phi_f1)))
# plt.plot(w_phi2, np.log10(np.abs(phi_f2)))
# # plt.plot(np.log10(phi_f1))
# # plt.plot(np.log10(phi_f2))
# # axes = plt.gca()
# # axes.set_xlim([0, 30])
# # plt.xlim([-1,60])
# plt.show()

a=np.nonzero(w_phi1>-cutoff)[0]
b=np.nonzero(w_phi1<cutoff)[-1]
print(a)
print(b)
a2=np.nonzero(w_phi2>-cutoff2)[0]
b2=np.nonzero(w_phi2<cutoff2)[-1]
print(a)
print(b)
# # plt.plot(w_phi1[a[0]:b[-1]],np.log10(phi_f1)[a[0]:b[-1]])
# # plt.plot(w_phi2[a2[0]:b2[-1]],np.log10(phi_f2)[a2[0]:b2[-1]])
# # plt.show()
phi_f1[b[-1]:]=0
phi_f1[:a[0]]=0
phi_f2[b2[-1]:]=0
phi_f2[:a2[0]]=0
cutphi_1=iFT((phi_f1))
cutphi_2=iFT((phi_f2))
#
# plt.plot(w_phi1, np.log10(phi_f1))
# plt.plot(w_phi2, np.log10(phi_f2))
# plt.xlim([0,100])
# plt.show()
#
# plt.plot(times,cutphi_1)
# plt.plot(times,cutphi_2)
# plt.show()


"""Doubling/Mirroring the spectrum first"""
d_phi_track=np.concatenate([phi_track,np.flip(phi_track)])
d_phi_track2=np.concatenate([phi_track2,np.flip(phi_track2)])
print(phi_track.shape)
print(d_phi_track.shape)
d_phi_track_shift=np.concatenate([phi_track_shift,np.flip(phi_track_shift)])
d_times = np.linspace(0.0, 2*cycles, len(d_phi_track))
plt.plot(d_times, d_phi_track)
plt.plot(d_times,d_phi_track_shift)
plt.show()

# cutoff=10
# cutoff2=cutoff

d_phi_f1=FT(d_phi_track_shift)
d_phi_f2=FT(d_phi_track2)
# phi_f1=np.fft.fftshift(np.fft.fft(phi_track))
# phi_f2=np.fft.fft(phi_track2)
wd_phi1= np.fft.fftshift(np.fft.fftfreq(len(d_phi_track),delta))
wd_phi1*= 2. * np.pi / prop_track.field

wd_phi2= np.fft.fftshift(np.fft.fftfreq(len(d_phi_track_shift),delta))
wd_phi2*= 2. * np.pi / prop_track2.field
plt.plot(wd_phi1, np.log10(np.abs(d_phi_f1)))
plt.plot(wd_phi2, np.log10(np.abs(d_phi_f2)))
# plt.plot(np.log10(phi_f1))
# plt.plot(np.log10(phi_f2))
# axes = plt.gca()
# axes.set_xlim([0, 30])
# plt.xlim([-1,60])
plt.show()

a=np.nonzero(wd_phi1>-cutoff)[0]
b=np.nonzero(wd_phi1<cutoff)[-1]
print(a)
print(b)
a2=np.nonzero(wd_phi2>-cutoff2)[0]
b2=np.nonzero(wd_phi2<cutoff2)[-1]
print(a)
print(b)
# plt.plot(w_phi1[a[0]:b[-1]],np.log10(phi_f1)[a[0]:b[-1]])
# plt.plot(w_phi2[a2[0]:b2[-1]],np.log10(phi_f2)[a2[0]:b2[-1]])
# plt.show()
d_phi_f1[b[-1]:]=0
d_phi_f1[:a[0]]=0
d_phi_f2[b2[-1]:]=0
d_phi_f2[:a2[0]]=0
d_cutphi_1=iFT((d_phi_f1))
d_cutphi_2=iFT((d_phi_f2))

plt.plot(wd_phi1, np.log10(d_phi_f1))
plt.plot(wd_phi2, np.log10(d_phi_f2))
plt.xlim([0,100])
plt.show()

plt.plot(wd_phi1, np.log10(d_phi_f1))
plt.plot(wd_phi2, np.log10(d_phi_f2))
plt.xlim([0,100])
plt.show()



plt.plot(times,d_cutphi_1[:len(times)])
plt.plot(times,phi_track_shift)
plt.show()

plt.plot(times,d_cutphi_2[:len(times)])
plt.plot(times,phi_track2)
plt.show()
# plt.plot(times,cutphi_2)
# plt.plot(times,phi_track2,linestyle='dashed')
# plt.show()

t1=0.52
t2=0.52
newcutphi=d_cutphi_1[:len(times)]
# newcutphi=d_cutphi_1[:len(times)]+2*np.pi*np.ones(len(times))
newcutphi2=d_cutphi_2[:len(times)]
# newcutphi=cutphi_1
# newcutphi2=cutphi_2
plt.plot(times,newcutphi)
plt.show()


#
#
# """Getting Breakdown time for D"""
gaps = []
list = []
corrs = []
Ulist = []
freqlist=[]
for j in range(0,1):
    f=7
    list.append(1 * f)
    U = 1 * f * t
    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=7*t, t=t, F0=F0, a=a, bc='pbc')
    Ulist.append(1*lat.U)
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
    #
    # gap = lat.U - 1* (2 - 4 * chem[0])
    chem_alt=lambda x: (16/lat.U)*np.sqrt(x**2-1)/(np.sinh(2*np.pi*x/lat.U))
    gap=scipy.integrate.quad(chem_alt, 1, np.inf, limit=240)[0]

    # gap = lat.U - 2* (2 - 4 * chem[0])

    gaps.append(gap)
    gaps.append(gap)
plt.plot(freqlist)
plt.show()

print(type(gaps))
print(type(corrs))
print(lat.a)
print(lat.F0)
print(lat.field)
breakdown = [a / (2*b) for a, b in zip(gaps, corrs)]
print(breakdown)
print(newcutphi)
breaktimes = []
for a in breakdown:
    for j in range(0,len(times)):
        c=newcutphi.real[j]
        if abs(c) > a:
            breaktimes.append(j)
            break
print(times[j])
print(j)
# print(newcutphi[breaktimes])

cutparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
    nx, cycles, U, t1, number, delta, field, F0, ascale,cutoff)
cutparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2,cutoff2)
np.save('./data/cutfreqs/phi'+cutparameternames,newcutphi)
# np.save('./data/cutfreqs/phi'+cutparameternames2,newcutphi2)



