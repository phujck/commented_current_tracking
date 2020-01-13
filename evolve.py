import numpy as np
from pyscf import fci
import harmonic


def ham1(lat, h, current_time, cycles):
    if lat.field == 0.:
        phi = 0.
    else:
        phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
            lat.field * current_time)
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * phi) * h_forwards + np.exp(-1.j * phi) * h_backwards


def f(lat, h1, psi):
    psi_r = psi.real
    psi_i = psi.imag
    h1_r = h1.real
    h1_i = h1.imag
    # H|psi>=(h1+h2)|psi>=(h1_r+ih1_i+h2)|psi>=(h1_r+ih1_i+h2)|psi_r>+i(h1_r+ih1_i+h2)|psi_i>
    pro = one_elec(lat, h1_r, psi_r) + 1.j * one_elec(lat, h1_i, psi_r, False) \
          + 1.j * one_elec(lat, h1_r, psi_i) - one_elec(lat, h1_i, psi_i, False) + two_elec(lat, psi_r, psi_i)
    return pro



def integrate_f(current_time,  psi, lat, cycles, h):
    ht = ham1(lat, h, current_time, cycles)
    return -1j * f(lat, ht, psi)





def one_elec(lat, h1, psi, sym=True):
    if sym:
        return fci.direct_spin1.contract_1e(h1, psi, lat.nsites, (lat.nup, lat.ndown))
    else:
        return fci.direct_nosym.contract_1e(h1, psi, lat.nsites, (lat.nup, lat.ndown))


def two_elec(lat, psi_r, psi_i):
    pro = 0.5 * fci.direct_uhf.contract_2e_hubbard((0, lat.U, 0), psi_r, lat.nsites, (lat.nup, lat.ndown)) \
          + 0.5 * 1.j * fci.direct_uhf.contract_2e_hubbard((0, lat.U, 0), psi_i, lat.nsites, (lat.nup, lat.ndown))
    return pro.flatten()


#
# def two_elec(lat, psi_r, psi_i):
#     pro =  fci.direct_uhf.contract_2e_hubbard((0, lat.U, 0), psi_r, lat.nsites, (lat.nup, lat.ndown)) \
#           +  1.j * fci.direct_uhf.contract_2e_hubbard((0, lat.U, 0), psi_i, lat.nsites, (lat.nup, lat.ndown))
#     return pro.flatten()


def RK4(lat, h, delta, current_time, psi, cycles):
    ht = ham1(lat, h, current_time, cycles)
    k1 = -1.j * delta * f(lat, ht, psi)
    ht = ham1(lat, h, current_time + 0.5 * delta, cycles)
    k2 = -1.j * delta * f(lat, ht, psi + 0.5 * k1)
    k3 = -1.j * delta * f(lat, ht, psi + 0.5 * k2)
    ht = ham1(lat, h, current_time + delta, cycles)
    k4 = -1.j * delta * f(lat, ht, psi + k3)
    return psi + (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def RK1(lat, h, delta, current_time, psi, cycles):
    ht = ham1(lat, h, current_time, cycles)
    k = -1.j * delta * f(lat, ht, psi)
    return psi + k


def phi_J_track(lat, current_time, J_reconstruct,neighbour, psi):
    # Import the current function
    # if current_time <self.delta:
    #     current=self.J_reconstruct(0)
    # else:
    #     current = self.J_reconstruct(current_time-self.delta)
    current = J_reconstruct(current_time)
    # Arrange psi to calculate the nearest neighbour expectations
    D = neighbour
    angle = np.angle(D)
    mag = np.abs(D)
    scalefactor = 2 * lat.a * lat.t * mag
    # assert np.abs(current)/scalefactor <=1, ('Current too large to reproduce, ration is %s' % np.abs(current/scalefactor))
    arg = -current / (2 * lat.a * lat.t * mag)
    phi = np.arcsin(arg + 0j) + angle
    # phi = np.arcsin(arg + 0j)
    return phi.real

def phi_reconstruct(lat, J_reconstruct,neighbour):
    # Import the current function
    # if current_time <self.delta:
    #     current=self.J_reconstruct(0)
    # else:
    #     current = self.J_reconstruct(current_time-self.delta)
    current = J_reconstruct
    # Arrange psi to calculate the nearest neighbour expectations
    D = neighbour
    angle = np.angle(D)
    mag = np.abs(D)
    scalefactor = 2 * lat.a * lat.t * mag
    # assert np.abs(current)/scalefactor <=1, ('Current too large to reproduce, ration is %s' % np.abs(current/scalefactor))
    arg = -current / (2 * lat.a * lat.t * mag)
    phi = np.arcsin(arg + 0j) + angle
    # phi = np.arcsin(arg + 0j)
    return phi.real


def phi_D_track(lat, current_time, D_reconstruct,two_body_expect, psi):
    # Import the current function
    # if current_time <self.delta:
    #     current=self.J_reconstruct(0)
    # else:
    #     current = self.J_reconstruct(current_time-self.delta)
    current=0
    # current = D_reconstruct(current_time)
    # if np.abs(current) < 1e-5:
    #     current =0
    # Arrange psi to calculate the nearest neighbour expectations
    D = -two_body_expect/lat.nsites
    angle = np.angle(D)
    mag = np.abs(D)
    scalefactor=2*lat.t*np.abs(D_reconstruct(0))
    arg = -current/ (2 * lat.t * mag)
    phi = np.arcsin(arg + 0j) + angle
    # phi = np.arcsin(arg + 0j)
    return phi.real



def ham_J_track(lat, h, current_time, J_reconstruct,neighbour, psi):
    phi=phi_J_track(lat, current_time, J_reconstruct,neighbour, psi)
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * phi) * h_forwards + np.exp(-1.j * phi) * h_backwards


def integrate_f_track_J(current_time,  psi, lat, h, J_reconstruct):
    ht = ham_J_track_ZVODE(current_time,psi,J_reconstruct,lat,h)
    return -1j * f(lat, ht, psi)

def integrate_f_cutfreqs(current_time,  psi, lat, h, phi_cut):
    ht = ham_J_cutfreqs_ZVODE(current_time,phi_cut,h)
    return -1j * f(lat, ht, psi)

def integrate_f_track_D(current_time,  psi, lat, h, J_reconstruct):
    ht = ham_D_track_ZVODE(current_time,psi,J_reconstruct,lat,h)
    return -1j * f(lat, ht, psi)

def ham_J_track_ZVODE(current_time,psi, J_reconstruct, lat, h):
    current = J_reconstruct(current_time)
    # Arrange psi to calculate the nearest neighbour expectations
    D = harmonic.nearest_neighbour_new(lat,h,psi)
    angle = np.angle(D)
    mag = np.abs(D)
    scalefactor = 2 * lat.a * lat.t * mag
    # assert np.abs(current)/scalefactor <=1, ('Current too large to reproduce, ration is %s' % np.abs(current/scalefactor))
    arg = -current / (2 * lat.a * lat.t * mag)
    phi = np.arcsin(arg + 0j) + angle
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * phi) * h_forwards + np.exp(-1.j * phi) * h_backwards


def ham_J_cutfreqs_ZVODE(current_time, phi_cut, h):
    phi = phi_cut(current_time)
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * phi) * h_forwards + np.exp(-1.j * phi) * h_backwards



def ham_D_track_ZVODE(current_time, psi, D_reconstruct,lat, h):
    # current = D_reconstruct(current_time)
    # if current_time < 0.2/lat.freq:
    #     current = 0
    current=0
    # Arrange psi to calculate the nearest neighbour expectations
    D = -harmonic.two_body_old (lat,psi)/ lat.nsites
    angle = np.angle(D)
    mag = np.abs(D)
    scalefactor = 2 * lat.t * np.abs(D_reconstruct(0))
    arg = -current / (2 * lat.t * mag)
    phi = np.arcsin(arg + 0j) + angle
    h_forwards = np.triu(h)
    h_forwards[0, -1] = 0.0
    h_forwards[-1, 0] = h[-1, 0]
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    return np.exp(1.j * phi) * h_forwards + np.exp(-1.j * phi) * h_backwards


def RK4_J_track(lat, h, delta, current_time, J_reconstruct, neighbour, psi):
    ht = ham_J_track(lat, h, current_time, J_reconstruct,neighbour, psi)
    k1 = -1.j * delta * f(lat, ht, psi)
    ht = ham_J_track(lat, h, current_time + 0.5 * delta,J_reconstruct,neighbour, psi)
    k2 = -1.j * delta * f(lat, ht, psi + 0.5 * k1)
    k3 = -1.j * delta * f(lat, ht, psi + 0.5 * k2)
    ht = ham_J_track(lat, h, current_time + delta, J_reconstruct,neighbour, psi)
    k4 = -1.j * delta * f(lat, ht, psi + k3)
    return psi + (k1 + 2. * k2 + 2. * k3 + k4) / 6.

def RK4_D_track(lat, h, delta, current_time, J_reconstruct, neighbour, psi):
    ht = ham_D_track(lat, h, current_time, J_reconstruct,neighbour, psi)
    k1 = -1.j * delta * f(lat, ht, psi)
    ht = ham_D_track(lat, h, current_time + 0.5 * delta,J_reconstruct,neighbour, psi)
    k2 = -1.j * delta * f(lat, ht, psi + 0.5 * k1)
    k3 = -1.j * delta * f(lat, ht, psi + 0.5 * k2)
    ht = ham_D_track(lat, h, current_time + delta, J_reconstruct,neighbour, psi)
    k4 = -1.j * delta * f(lat, ht, psi + k3)
    return psi + (k1 + 2. * k2 + 2. * k3 + k4) / 6.

