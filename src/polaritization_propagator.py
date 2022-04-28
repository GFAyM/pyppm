from pyscf import gto, scf
from pyscf.gto import Mole
import numpy 
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.dft import numint
from pyscf.data import nist

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

        m = a + b
        m = m.reshape(nocc*nvir,nocc*nvir, order='C')
        
        return m

    def h1_fc(self,atmlst):

        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        mol = self.mf.mol
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:,mo_occ> 0]
        orbv = mo[:,mo_occ==0]
        fac = 8*numpy.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
        h1 = []
        for ia in atmlst:
            h1.append(fac * numpy.einsum('p,i->pi', orbv[ia], orbo[ia]))
        return h1

    def uniq_atoms(self, nuc_pair):
        atm1lst = sorted(set([i for i,j in nuc_pair]))
        atm2lst = sorted(set([j for i,j in nuc_pair]))
        atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
        atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
        return atm1dic, atm2dic


    @property
    def pol_prop(self):
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        mol = self.mf.mol
        nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        atm1dic, atm2dic = self.uniq_atoms(nuc_pair=nuc_pair)

        h2 = self.h1_fc(sorted(atm2dic.keys()))
        h1 = self.h1_fc(sorted(atm1dic.keys()))    
        m = self.m_matrix_triplet
        half_prop = h1*m
        para = []
        for i,j in nuc_pair:
            at1 = atm1dic[i]
            at2 = atm2dic[j]
            e = numpy.einsum('ij,ij', h2[at2], half_prop[at1])
            para.append(e*4)  # *4 for +c.c. and for double occupancy
        return numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))

    @property
    def kernel(self):
        mol = self.mf.mol
        nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        atm1dic, atm2dic = self.uniq_atoms(nuc_pair=nuc_pair)

        h2 = self.h1_fc(sorted(atm2dic.keys()))
        return h2