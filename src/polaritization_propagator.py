from pyscf import gto, scf
from pyscf.gto import Mole
import numpy 
from pyscf import lib
import attr
from pyscf import ao2mo

@attr.s
class Prop_pol:
    """_summary_
    """
    mf = attr.ib(default=None, type=scf.hf.RHF, validator=attr.validators.instance_of(scf.hf.RHF))
    #mo_energy = attr.ib(default=None, type=numpy.ndarray, validator=attr.validators.instance_of(numpy.ndarray))
    #mo_coeff = attr.ib(default=None, type=numpy.ndarray, validator=attr.validators.instance_of(numpy.ndarray))
    #mo_occ = attr.ib(default=None, type=numpy.ndarray, validator=attr.validators.instance_of(numpy.ndarray))

        
        
    @property
    def m_matrix_triplet(self):
        r'''A and B matrices for TDDFT response function.

        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        '''

        mol = self.mf.mol
        mo_energy = self.mf.mo_energy
        mo_coeff = self.mf.mo_coeff
        nao, nmo = mo_coeff.shape
        mo_occ = self.mf.mo_occ
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]
        nvir = orbv.shape[1]
        nocc = orbo.shape[1]
        mo = numpy.hstack((orbo,orbv))
        nmo = nocc + nvir


        e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
        a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
        b = numpy.zeros_like(a)

        eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:])
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])

        m = a+b
        m = m.reshape(nocc*nvir,nocc*nvir)
        
        return m
