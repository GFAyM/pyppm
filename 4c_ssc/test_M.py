from pyscf import gto, scf, ao2mo
import numpy
from pyscf import lib

def M(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    n2c = nmo // 2
    #occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    #viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    occidx = numpy.where(mo_occ == 1)[0]
    viridx = numpy.where(mo_occ == 0)[0]
        
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    nmo = nocc + nvir
    mo = numpy.hstack((orbo, orbv))
    c1 = .5 / lib.param.LIGHT_SPEED
    #moL = numpy.asarray(mo[:n2c], order='F')
    #moS = numpy.asarray(mo[n2c:], order='F') * c1
    moL = mo[:n2c,:]
    moS = mo[n2c:,:] * c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]

    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    eri_mo = ao2mo.kernel(mol, [orboL, moL, moL, moL], intor='int2e_spinor')
    eri_mo+= ao2mo.kernel(mol, [orboS, moS, moS, moS], intor='int2e_spsp1spsp2_spinor')
    eri_mo+= ao2mo.kernel(mol, [orboS, moS, moL, moL], intor='int2e_spsp1_spinor')
    eri_mo+= ao2mo.kernel(mol, [moS, moS, orboL, moL], intor='int2e_spsp1_spinor').T
    eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)

    a_ = numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
    a = a + a_
    a__ = numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:])
    a = a - a__
    b_ = numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
    b = b + b_
    b__ = numpy.einsum('jbia->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
    b = b -  b__
    m = a+b
    m = m.reshape(nocc*nvir,nocc*nvir, order='C')
    #p = numpy.linalg.inv(m)
    return b_, b__



mol = gto.M(atom='''
O        0.0000000000            0.0000000000           -0.0409868122
H1       0.0000000000            0.7567917171            0.5640254210
H2       0.0000000000           -0.7567917171            0.5640254210
''', basis='ccpvdz', unit='angstrom')

mf = scf.DHF(mol).run()
mf.scf()

b1,b2 = M(mf)

