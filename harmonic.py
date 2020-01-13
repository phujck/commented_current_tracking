import numpy as np
import evolve as evolve
from scipy import signal
from pyscf import fci
import definition as harmonic

def spectrum_welch(at,delta):
    return signal.welch(at,1./delta,nperseg=len(at),scaling='spectrum')

def spectrum(at,delta): 
    w = np.fft.rfftfreq(len(at),delta)
    spec = 2.*(abs(np.fft.rfft(at))**2.)/(len(at))**2.
    return (w, spec)

def spectrum_hanning(at,delta):
    w = np.fft.rfftfreq(len(at),delta)
    win = np.hanning(len(at))
    spec = 2.*(abs(np.fft.rfft(win*at))**2.)/(sum(win**2.))**2
    return (w, spec)

def current(lat,h,current_time,cycles):
    if lat.field==0.:
        phi = 0.
    else:
        phi = (lat.a*lat.F0/lat.field)*(np.sin(lat.field*current_time/(2.*cycles))**2.)*np.sin(lat.field*current_time)
    h_forwards = np.triu(h)
    h_forwards[0,-1] = 0.0
    h_forwards[-1,0] = h[-1,0]
    h_backwards = np.tril(h)
    h_backwards[-1,0] = 0.0
    h_backwards[0,-1] = h[0,-1]
    return 1.j*lat.a*(np.exp(-1.j*phi)*h_backwards - np.exp(1.j*phi)*h_forwards)

def current_track(lat,h,phi):
    h_forwards = np.triu(h)
    h_forwards[0,-1] = 0.0
    h_forwards[-1,0] = h[-1,0]
    h_backwards = np.tril(h)
    h_backwards[-1,0] = 0.0
    h_backwards[0,-1] = h[0,-1]
    return 1.j*lat.a*(np.exp(-1.j*phi)*h_backwards - np.exp(1.j*phi)*h_forwards)

def fJ(lat,J,psi):
    psi_r = psi.real
    psi_i = psi.imag 
    Jr = J.real
    Ji = J.imag
    #J|psi>=(J_r+iJ_i)|psi>=(J_r+iJ_i)|psi_r>+i(J_r+iJ_i)|psi_i>
    pro = evolve.one_elec(lat,Jr,psi_r) + 1j*evolve.one_elec(lat,Ji,psi_r,False) + 1j*evolve.one_elec(lat,Jr,psi_i) \
    - evolve.one_elec(lat,Ji,psi_i,False)
    return pro

def J_expectation_track(lat,h,psi,phi):
    J = current_track(lat,h,phi)
    # print('real part')
    # print(J.real)
    # print('imaginary part')
    # print(J.imag)
    # print((np.dot(psi.conj(),fJ(lat,J,psi))).imag)
    return (np.dot(psi.conj(),fJ(lat,J,psi))).real


def J_expectation(lat,h,psi,current_time,cycles):
    J = current(lat,h,current_time,cycles)
    # print('real part')
    # print(J.real)
    # print('imaginary part')
    # print(J.imag)
    # print((np.dot(psi.conj(),fJ(lat,J,psi))).imag)
    return (np.dot(psi.conj(),fJ(lat,J,psi))).real

def J_expectation_cutfreq(lat,h,psi, cut_phi):
    J = current_cutfreq(lat,h, cut_phi)
    # print('real part')
    # print(J.real)
    # print('imaginary part')
    # print(J.imag)
    # print((np.dot(psi.conj(),fJ(lat,J,psi))).imag)
    return (np.dot(psi.conj(),fJ(lat,J,psi))).real

def current_cutfreq(lat,h,cut_phi):
    if lat.field==0.:
        phi = 0.
    else:
        phi = cut_phi
    h_forwards = np.triu(h)
    h_forwards[0,-1] = 0.0
    h_forwards[-1,0] = h[-1,0]
    h_backwards = np.tril(h)
    h_backwards[-1,0] = 0.0
    h_backwards[0,-1] = h[0,-1]
    return 1.j*lat.a*(np.exp(-1.j*phi)*h_backwards - np.exp(1.j*phi)*h_forwards)


def nearest_neighbour(lat, psi):
    psi = np.reshape(psi, (fci.cistring.num_strings(lat.nsites, lat.nup), fci.cistring.num_strings(lat.nsites, lat.ndown)))
    D = 0.
    for j in [0, 1]:
        for i in range(lat.nsites - 1):
            D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [i, i + 1], [1, 0], [j, j])
        # Assuming periodic conditions
        D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [lat.nsites - 1, 0], [1, 0], [j, j])
    return D.conj()





# calculates <c^*_jc_j+1> using the pyscf contraction
def nearest_neighbour_new(lat,h, psi):
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    h_forwards = np.triu(h)
    h_forwards[0,-1] = 0.0
    h_forwards[-1,0] = h[-1,0]
    psi_r = psi.real
    psi_i = psi.imag
    hr = -h_backwards.real
    hi = -h_backwards.imag
    # J|psi>=(J_r+iJ_i)|psi>=(J_r+iJ_i)|psi_r>+i(J_r+iJ_i)|psi_i>
    pro = evolve.one_elec(lat, hr, psi_r,False) + 1j*evolve.one_elec(lat,hi,psi_r,False)+ 1j*evolve.one_elec(lat,hr,psi_i,False) \
    - evolve.one_elec(lat,hi,psi_i,False)
    return np.dot(psi.conj(), pro)
    # return (np.dot(psi, pro))

# calculates <c^*_j+1c_j> using the pyscf contraction
def nearest_neighbour_new_2(lat,h, psi):
    h_backwards = np.tril(h)
    h_backwards[-1, 0] = 0.0
    h_backwards[0, -1] = h[0, -1]
    h_forwards = np.triu(h)
    h_forwards[0,-1] = 0.0
    h_forwards[-1,0] = h[-1,0]
    psi_r = psi.real
    psi_i = psi.imag
    hr = -h_forwards.real
    hi = -h_forwards.imag
    # J|psi>=(J_r+iJ_i)|psi>=(J_r+iJ_i)|psi_r>+i(J_r+iJ_i)|psi_i>
    pro = evolve.one_elec(lat, hr, psi_r,False) + 1j*evolve.one_elec(lat,hi,psi_r,False)+ 1j*evolve.one_elec(lat,hr,psi_i,False) \
    - evolve.one_elec(lat,hi,psi_i,False)
    return np.dot(pro.conj(),psi)
    # return np.conj((np.dot(psi,pro)))

def phi(lat,current_time,cycles):
    if lat.field==0.:
        return 0.
    else:
        return (lat.a*lat.F0/lat.field)*(np.sin(lat.field*current_time/(2.*cycles))**2.)*np.sin(lat.field*current_time)
        
def J_expectation2(lat,h,psi,current_time,cycles,neighbour):
    conjugator = np.exp(-1.j * phi(lat,current_time,cycles)) * neighbour
    return (-1.j * lat.a * lat.t * (conjugator - np.conj(conjugator))).real

def J_expectation3(lat,h,psi,current_time,cycles,neighbour_1,neighbour_2):
    first = np.exp(-1.j * phi(lat,current_time,cycles)) * neighbour_1
    second=np.exp(1.j * phi(lat,current_time,cycles)) * neighbour_2.conj()
    return (-1.j * lat.a * (first-second)).real


def two_body_old(sys, psi):
    """Contribution from two-body-terms commutator with c*_k c_k+1"""
    # psi = np.reshape(psi,
    #                  (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for i in range(sys.nsites):
        w = (i + 1) % sys.nsites
        v = (i - 1) % sys.nsites

        D += harmonic.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, w, i, i], [1, 0, 1, 0], [1, 1, 0, 0])

        D += harmonic.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, i, w], [1, 0, 1, 0], [1, 1, 0, 0])

        D -= harmonic.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [v, i, i, i], [1, 0, 1, 0], [1, 1, 0, 0])

        D -= harmonic.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i, v, i], [1, 0, 1, 0], [1, 1, 0, 0])

    return D.conj()

def one_elec(lat,h1,psi,sym=True):
    if sym:
        return fci.direct_spin1.contract_1e(h1,psi,lat.nsites,(lat.nup,lat.ndown))
    else:
        return fci.direct_nosym.contract_1e(h1,psi,lat.nsites,(lat.nup,lat.ndown))

def two_elec(lat,psi_r,psi_i):
    pro = 0.5*fci.direct_uhf.contract_2e_hubbard((0,lat.U,0),psi_r,lat.nsites,(lat.nup,lat.ndown)) \
    + 0.5*1.j*fci.direct_uhf.contract_2e_hubbard((0,lat.U,0),psi_i,lat.nsites,(lat.nup,lat.ndown))
    return pro.flatten()

def two_body(lat, h, psi_r, psi_i):
    """Commutator of two-body Hubbard term with c*_k c_k+1, including factor of U"""

    #minus sign is to eliminate the hopping constant
    h_backwards = -np.tril(h)
    h_backwards[0, -1] = h[0, -1]
    h_backwards[-1, 0] = 0.
    #calculates the first expectation value in the commutator, contracting from right to left
    psi_new = one_elec(lat, h_backwards, psi_r, False) + 1.j*one_elec(lat, h_backwards, psi_i, False)
    psi_new = two_elec(lat, psi_new.real, psi_new.imag)
    expectation1 = np.dot((psi_r+1j*psi_i).conj(), psi_new)

    h_forwards = -np.triu(h)
    h_forwards[-1, 0] = h[-1, 0]
    h_forwards[0, -1] = 0.
    #calculates the second expectation value in the commutator, contracting from right to left
    psi_new = two_elec(lat, psi_r, psi_i)
    psi_new = one_elec(lat, h_backwards, psi_new.real, False) + 1.j*one_elec(lat, h_backwards, psi_new.imag, False)
    expectation2 = np.dot((psi_r+1j*psi_i).conj(), psi_new)

    return expectation1 - expectation2



# These expectations are used to compute the energy gap between the system and system + a doublon pair

# one body energies

def one_energy(lat,psi, phi):
    D=0
    for j in [0, 1]:
        for i in range(lat.nsites - 1):
            D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [i, i + 1], [1, 0], [j, j])
        # Assuming periodic conditions
        D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [lat.nsites - 1, 0], [1, 0], [j, j])
    one_e=-2*lat.t*np.real(D.conj()*np.exp(-1j*phi))
    return one_e
#


# alternate method for calculating the one body energy by directly adding extra electrons to expectation.

# def doublon_one_energy(lat,psi, phi):
#     D=0
#     for k in range(lat.nsites):
#         for j in [0, 1]:
#             for i in range(lat.nsites - 1):
#                 D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [k,k,i, i + 1,k,k], [0,0,1,0,1,1], [1,0,j, j,0,1])
#             # Assuming periodic conditions
#             D += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [lat.nsites - 1, 0], [1, 0], [j, j])
#         one_e=-2*lat.t*np.real(D.conj()*np.exp(-1j*phi))/lat.nsites
#     return one_e


# this method uses a compute inner product after adding two electrons to the wavefunction.

# averaging over L^2:


def doublon_one_energy(lat,psi, phi):
    D=0
    for j in [0, 1]:
        for i in range(lat.nsites - 1):
            D += harmonic.compute_inner_product_doublon_mix(psi, lat.nsites, (lat.nup, lat.ndown), [i, i + 1], [1, 0], [j, j])
        # Assuming periodic conditions
        D += harmonic.compute_inner_product_doublon_mix(psi, lat.nsites, (lat.nup, lat.ndown), [lat.nsites - 1, 0], [1, 0], [j, j])
    one_e=-2*lat.t*np.real(D.conj()*np.exp(-1j*phi))
    return one_e

# Averaging over L:
# def doublon_one_energy(lat,psi, phi):
#     D=0
#     for j in [0, 1]:
#         for i in range(lat.nsites - 1):
#             D += harmonic.compute_inner_product_doublon(psi, lat.nsites, (lat.nup, lat.ndown), [i, i + 1], [1, 0], [j, j])
#         # Assuming periodic conditions
#         D += harmonic.compute_inner_product_doublon(psi, lat.nsites, (lat.nup, lat.ndown), [lat.nsites - 1, 0], [1, 0], [j, j])
#     one_e=-2*lat.t*np.real(D.conj()*np.exp(-1j*phi))
#     return one_e


def two_energy(lat,psi):
    two_e=0
    for i in range(lat.nsites):
        two_e += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [i, i, i, i], [1, 0, 1, 0], [1, 1, 0, 0])
        # Assuming periodic conditions
    return lat.U*two_e

# ditto alternate two body energy


# def doublon_two_energy(lat,psi):
#     two_e=0
#     for k in range(lat.nsites):
#         for i in range(lat.nsites):
#             two_e += harmonic.compute_inner_product(psi, lat.nsites, (lat.nup, lat.ndown), [k,k,i, i, i, i,k,k], [0,0,1,0, 1, 0,1,1], [1,0,1, 1, 0, 0,0,1])
#             # Assuming periodic conditions
#         return lat.U*two_e


# this one for adding electrons averaged over L^2
def doublon_two_energy(lat,psi):
    two_e=0
    for i in range(lat.nsites):
        two_e += harmonic.compute_inner_product_doublon_mix(psi, lat.nsites, (lat.nup, lat.ndown), [i, i, i, i], [1, 0, 1, 0], [1, 1, 0, 0])
        # Assuming periodic conditions
    return lat.U*two_e

# This one for averaging over doublon additions (i.e. over L)
# def doublon_two_energy(lat,psi):
#     two_e=0
#     for i in range(lat.nsites):
#         two_e += harmonic.compute_inner_product_doublon(psi, lat.nsites, (lat.nup, lat.ndown), [i, i, i, i], [1, 0, 1, 0], [1, 1, 0, 0])
#         # Assuming periodic conditions
#     return lat.U*two_e


def singlon_energy(lat,psi, phi):
    two_e = 0
    for i in range(lat.nsites):
        two_e += harmonic.compute_inner_product_singlon(psi, lat.nsites, (lat.nup, lat.ndown), [i, i, i, i],
                                                            [1, 0, 1, 0], [1, 1, 0, 0])
    D = 0
    for j in [0, 1]:
        for i in range(lat.nsites - 1):
            D += harmonic.compute_inner_product_singlon(psi, lat.nsites, (lat.nup, lat.ndown), [i, i + 1], [1, 0],
                                                            [j, j])
        # Assuming periodic conditions
        D += harmonic.compute_inner_product_singlon(psi, lat.nsites, (lat.nup, lat.ndown), [lat.nsites - 1, 0],
                                                        [1, 0], [j, j])
    one_e = -2 * lat.t * np.real(D.conj() * np.exp(-1j * phi))
    return -2*lat.t*np.real(D.conj()*np.exp(-1j*phi))+ lat.U * two_e