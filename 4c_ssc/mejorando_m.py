from pyscf import gto, scf
from pyscf.gto import Mole
import numpy 
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyscf import tools
from pyscf.lib import logger
import scipy
from pyscf.lib import current_memory


mol_h2s = gto.M(atom='''
S      .0000000000        0.0000000000        -.2249058930
H1   -1.4523499293         .0000000000         .8996235720
H2    1.4523499293         .0000000000         .8996235720
''', basis='cc-pvdz', unit='bhor', verbose=3)

rhf = scf.DHF(mol_h2s).run()



def M(mol, mo_coeff, mo_occ, mo_energy):
    """
    A and B matrices for TDDFT response function.
    

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    This matrices was extracted from the tdscf pyscf module.
    
    Returns:

        Numpy.array: Inverse of Principal Propagator 
    """

    n4c, nmo = mo_coeff.shape
    n2c = nmo // 2


    occidx = numpy.where(mo_occ == 1)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    orbo = mo_coeff[:,mo_occ==1]
    orbv = mo_coeff[:,mo_occ==0]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    c1 = .5 / lib.param.LIGHT_SPEED
    mo = numpy.hstack((orbo, orbv))
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F')* c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]
    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)
    
    
    eri_mo = ao2mo.general(mol, [orboL, moL, moL, moL], intor='int2e_spinor')
    eri_mo += ao2mo.general(mol, [orboS, moS, moS, moS], intor='int2e_spsp1spsp2_spinor')
    eri_mo+= ao2mo.general(mol, [orboS, moS, moL, moL], intor='int2e_spsp1_spinor')
    eri_mo+= ao2mo.general(mol, [moS, moS, orboL, moL], intor='int2e_spsp1_spinor').T
    eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
    
    a = a + lib.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
    a = a - lib.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:])
    b = b + lib.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
    b = b - lib.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])

    m = a+b
    m = m.reshape(nocc*nvir,nocc*nvir, order='C')
    m_y = a-b
    m_y = m_y.reshape(nocc*nvir,nocc*nvir, order='C')
    return m, m_y

def A1(mol, mo_coeff, mo_occ):
    """
    A and B matrices for TDDFT response function.
    

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    This matrices was extracted from the tdscf pyscf module.
    
    Returns:

        Numpy.array: Inverse of Principal Propagator 
    """

    n4c, nmo = mo_coeff.shape
    n2c = nmo // 2


    orbo = mo_coeff[:,mo_occ==1]
    orbv = mo_coeff[:,mo_occ==0]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    c1 = .5 / lib.param.LIGHT_SPEED
    mo = numpy.hstack((orbo, orbv))
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F')* c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]
    orbvL = moL[:,nocc:]
    orbvS = moS[:,nocc:] 

    
    print("total memory: %.1f MiB" % current_memory()[0])
    a1 = ao2mo.general(mol, [orboL, orbvL, orbvL, orboL], intor='int2e_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    a1 += ao2mo.general(mol, [orboS, orbvS, orbvS, orboS], intor='int2e_spsp1spsp2_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    a1+= ao2mo.general(mol, [orboS, orbvS, orbvL, orboL], intor='int2e_spsp1_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    a1+= ao2mo.general(mol, [orbvS, orboS, orboL, orbvL], intor='int2e_spsp1_spinor').T
    a1 = a1.reshape(nocc,nvir,nvir,nocc)
    print("total memory: %.1f MiB" % current_memory()[0])
    a1 = lib.einsum('iabj->iajb', a1)

    return a1
def A2(mol, mo_coeff, mo_occ):
    """
    A and B matrices for TDDFT response function.
    

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    This matrices was extracted from the tdscf pyscf module.
    
    Returns:

        Numpy.array: Inverse of Principal Propagator 
    """

    n4c, nmo = mo_coeff.shape
    n2c = nmo // 2


    orbo = mo_coeff[:,mo_occ==1]
    orbv = mo_coeff[:,mo_occ==0]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    c1 = .5 / lib.param.LIGHT_SPEED
    mo = numpy.hstack((orbo, orbv))
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F')* c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]
    orbvL = moL[:,nocc:]
    orbvS = moS[:,nocc:] 

    
    print("total memory: %.1f MiB" % current_memory()[0])
    a2 = ao2mo.general(mol, [orboL, orboL, orbvL, orbvL], intor='int2e_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    a2 += ao2mo.general(mol, [orboS, orboS, orbvS, orbvS], intor='int2e_spsp1spsp2_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    a2+= ao2mo.general(mol, [orboS, orboS, orbvL, orbvL], intor='int2e_spsp1_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    a2+= ao2mo.general(mol, [orbvS, orbvS, orboL, orboL], intor='int2e_spsp1_spinor').T
    a2 = a2.reshape(nocc,nocc,nvir,nvir)
    print("total memory: %.1f MiB" % current_memory()[0])
    a2 = lib.einsum('iabj->iajb', a2)

    a2 = lib.einsum('ijba->iajb', a2)
    print(nocc*nocc*nvir*nvir)
    return a2


def B(mol, mo_coeff, mo_occ):
    """
    A and B matrices for TDDFT response function.
    

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)

    This matrices was extracted from the tdscf pyscf module.
    
    Returns:

        Numpy.array: Inverse of Principal Propagator 
    """

    n4c, nmo = mo_coeff.shape
    n2c = nmo // 2


    orbo = mo_coeff[:,mo_occ==1]
    orbv = mo_coeff[:,mo_occ==0]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    c1 = .5 / lib.param.LIGHT_SPEED
    mo = numpy.hstack((orbo, orbv))
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F')* c1
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]
    orbvL = moL[:,nocc:]
    orbvS = moS[:,nocc:] 

    
    print("total memory: %.1f MiB" % current_memory()[0])
    b_ = ao2mo.general(mol, [orboL, orbvL, orboL, orbvL], intor='int2e_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    b_ += ao2mo.general(mol, [orboS, orbvS, orboS, orbvS], intor='int2e_spsp1spsp2_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    b_+= ao2mo.general(mol, [orboS, orbvS, orboL, orbvL], intor='int2e_spsp1_spinor')
    print("total memory: %.1f MiB" % current_memory()[0])
    b_+= ao2mo.general(mol, [orboS, orbvS, orboL, orbvL], intor='int2e_spsp1_spinor').T
    b_ = b_.reshape(nocc,nvir,nocc,nvir)
    print("total memory: %.1f MiB" % current_memory()[0])
    b_ =  lib.einsum('iajb->iajb', b_)
    b_ -= lib.einsum('jaib->iajb', b_)
    print(nocc*nmo*nmo*nmo) 
    return b_


#m, my = M(mol_h2s, rhf.mo_coeff, rhf.mo_occ, rhf.mo_energy)

#a1 = A1(mol_h2s, rhf.mo_coeff, rhf.mo_occ, rhf.mo_energy)
#print(a1.size)

#a2 = A2(mol_h2s, rhf.mo_coeff, rhf.mo_occ, rhf.mo_energy)
#print(a2.size)

b = B(mol_h2s, rhf.mo_coeff, rhf.mo_occ)
print(b.size)