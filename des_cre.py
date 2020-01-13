import numpy
from pyscf import lib
from pyscf.fci import cistring


def des_a(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N-1)-electron wavefunction by removing an alpha electron from
    the N-electron wavefunction.

    ... math::

        |N-1\rangle = \hat{a}_p |N\rangle

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the annihilation operator

    Returns:
        2D array, row for alpha strings and column for beta strings.  Note it
        has different number of rows to the input CI coefficients
    '''
    neleca, nelecb = neleca_nelecb
    if neleca <= 0:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    des_index = cistring.gen_des_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca-1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]),dtype=ci0[0,0].__class__)

    entry_has_ap = (des_index[:,:,1] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3].astype(complex)
    #print(addr_ci0)
    #print(addr_ci1)
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

def des_b(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N-1)-electron wavefunction by removing a beta electron from
    N-electron wavefunction.

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the annihilation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of columns to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if nelecb <= 0:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    des_index = cistring.gen_des_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb-1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1),dtype=ci0[0,0].__class__)

    entry_has_ap = (des_index[:,:,1] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3].astype(complex)
    # This sign prefactor accounts for interchange of operators with alpha and beta spins
    if neleca % 2 == 1:
        sign *= -1
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1

def cre_a(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N+1)-electron wavefunction by adding an alpha electron in
    the N-electron wavefunction.

    ... math::

        |N+1\rangle = \hat{a}^+_p |N\rangle

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the creation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of rows to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if neleca >= norb:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    cre_index = cistring.gen_cre_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca+1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]),dtype=ci0[0,0].__class__)

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3].astype(complex)
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

# construct (N+1)-electron wavefunction by adding a beta electron to
# N-electron wavefunction:
def cre_b(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N+1)-electron wavefunction by adding a beta electron in
    the N-electron wavefunction.

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the creation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of columns to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if nelecb >= norb:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    cre_index = cistring.gen_cre_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb+1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1),dtype=ci0[0,0].__class__)

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3].astype(complex)
    # This sign prefactor accounts for interchange of operators with alpha and beta spins
    if neleca % 2 == 1:
        sign *= -1
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1