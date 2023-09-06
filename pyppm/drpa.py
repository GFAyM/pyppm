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
from pyppm.rpa import RPA
from pyscf.lib import current_memory

def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

@attr.s
class DRPA(RPA):
    """ This class Calculates the J-coupling between two nuclei in the 4-component framework

        Need, as attribute, a DHF object

    Returns:
        _type_: _description_
    """ 
    mf = attr.ib(default=None, type=scf.dhf.DHF, validator=attr.validators.instance_of(scf.dhf.DHF))
    
        
    @property
    def M(self):
        """
        A and B matrices for TDDFT response function.
        

        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        This matrices was extracted from the tdscf pyscf module.
        
        Returns:

            Numpy.array: Inverse of Principal Propagator 
        """
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        n4c, nmo = mo_coeff.shape
        n2c = nmo // 2

        mo_energy = self.mf.mo_energy
        occidx = numpy.where(mo_occ == 1)[0]
        viridx = numpy.where(mo_occ == 0)[0]
        orbo = mo_coeff[:,mo_occ==1]
        orbv = mo_coeff[:,mo_occ==0]
        nvir = orbv.shape[1]
        nocc = orbo.shape[1]
        c1 = .5 / lib.param.LIGHT_SPEED
        mo = numpy.hstack((orbo, orbv))
        moL = numpy.asarray(mo[:n2c], order='F')
        moS = numpy.asarray(mo[n2c:], order='F') * c1
        orboL = moL[:,:nocc]
        orboS = moS[:,:nocc]
        e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
        a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
        b = numpy.zeros_like(a)

        eri_mo = ao2mo.kernel(mol, [orboL.conj(), moL.conj(), moL.conj(), moL.conj()], intor='int2e_spinor')        
        eri_mo_b1 = ao2mo.kernel(mol, [orboL.conj(), moL.conj(), moL, moL], intor='int2e_spinor')
        eri_mo_b2 = ao2mo.kernel(mol, [orboL, moL.conj(), moL.conj(), moL], intor='int2e_spinor')
        eri_mo+= ao2mo.kernel(mol, [orboS.conj(), moS.conj(), moS.conj(), moS.conj()], intor='int2e_spsp1spsp2_spinor')
        eri_mo_b1+= ao2mo.kernel(mol, [orboS.conj(), moS.conj(), moS, moS], intor='int2e_spsp1spsp2_spinor')
        eri_mo_b2+= ao2mo.kernel(mol, [orboS, moS.conj(), moS.conj(), moS], intor='int2e_spsp1spsp2_spinor')
        
        eri_mo+= ao2mo.kernel(mol, [orboS.conj(), moS.conj(), moL.conj(), moL.conj()], intor='int2e_spsp1_spinor')
        eri_mo+= ao2mo.kernel(mol, [moS.conj(), moS.conj(), orboL.conj(), moL.conj()], intor='int2e_spsp1_spinor').T
        
        eri_mo_b1+= ao2mo.kernel(mol, [orboS.conj(), moS.conj(), moL, moL], intor='int2e_spsp1_spinor')
        eri_mo_b1+= ao2mo.kernel(mol, [moS, moS, orboL.conj(), moL.conj()], intor='int2e_spsp1_spinor').T
        
        eri_mo_b2+= ao2mo.kernel(mol, [orboS, moS.conj(), moL.conj(), moL], intor='int2e_spsp1_spinor')
        
        eri_mo_b2+= ao2mo.kernel(mol, [moS.conj(), moS, orboL, moL.conj()], intor='int2e_spsp1_spinor').T
        
        #eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)

        a = a + numpy.einsum('iabj->iajb', eri_mo.reshape(nocc,nmo,nmo,nmo)
                             [:nocc,nocc:,nocc:,:nocc])
        a = a - numpy.einsum('ijba->iajb', eri_mo.reshape(nocc,nmo,nmo,nmo)
                             [:nocc,:nocc,nocc:,nocc:])
        b = b + numpy.einsum('iajb->iajb', eri_mo_b1.reshape(nocc,nmo,nmo,nmo)
                             [:nocc,nocc:,:nocc,nocc:])
        b = b - numpy.einsum('jaib->iajb', eri_mo_b2.reshape(nocc,nmo,nmo,nmo)
                             [:nocc,nocc:,:nocc,nocc:])
        
        m = a + b 
        m = m.reshape(nocc*nvir,nocc*nvir, order='C')
        m_y = a - b
        m_y = m_y.reshape(nocc*nvir,nocc*nvir, order='C')
        return m, m_y


    def make_perturbator(self, atmlst):
        """
        Perturbator in 4component framework
        Extracted from properties module, ssc/dhf.py

        Args:
            atmlst (list): The atom number in which is centered the 
            perturbator

        Returns:
            list: list with perturbator in molecular basis
        """
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        orbo = mo_coeff[:,mo_occ==1]
        orbv = mo_coeff[:,mo_occ==0]
        n4c = mo_coeff.shape[0]
        n2c = n4c // 2
        h1 = []
        for ia in atmlst:
            mol.set_rinv_origin(mol.atom_coord(ia))
            a01int = mol.intor('int1e_sa01sp_spinor', 3)
            h01 = numpy.zeros((n4c,n4c), numpy.complex128)
            for k in range(3):
                h01[:n2c,n2c:] = .5 * a01int[k]
                h01[n2c:,:n2c] = .5 * a01int[k].conj().T
                h1.append(orbv.conj().T.dot(h01).dot(orbo))
        return h1

    def pp_ssc_4c(self, atm1lst, atm2lst):
        """In this Function generate de Response << ; >>, i.e, multiplicate the perturbators centered
        in all nuclei with the principal propagator.


        Returns:
            numpy.array: J response
        """
        
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        orbo = mo_coeff[:,mo_occ==1]
        orbv = mo_coeff[:,mo_occ==0]
        nvir = orbv.shape[1]
        nocc = orbo.shape[1]


        h1 = numpy.asarray(self.make_perturbator(atm1lst)).reshape(1,3,nvir,nocc)[0]        
        
        h2 = numpy.asarray(self.make_perturbator(atm2lst)).reshape(1,3,nvir,nocc)[0]
        m_xz, m_y = self.M

        p_xz = -scipy.linalg.inv(m_xz)

        p_xz = p_xz.reshape(nocc,nvir,nocc,nvir)

        p_y = -scipy.linalg.inv(m_y)
        
        p_y = p_y.reshape(nocc,nvir,nocc,nvir)
        para = []
        para_y = []

        #for i,j in nuc_pair:
        #e1 = numpy.einsum('iajb,ybj->yai', p_xz, h2) 
        #print(e1)
        #e = numpy.einsum('xai,yai->xy', h1, e1.conj()) * 2 

        e = numpy.einsum('xai,iajb,ybj->xy', h1, p_xz.conj(), h2.conj()) * 2
        para.append(e.real)
        #e1 = numpy.einsum('iajb,ybj->yai', p_y, h2) 
        #e_y = numpy.einsum('xai,yai->xy', h1, e1.conj()) * 2 

        e_y = numpy.einsum('xai,iajb,ybj->xy', h1, p_y.conj(), h2.conj()) * 2
        para_y.append(e_y.real)
        
        resp = numpy.asarray(para)
        resp_y = numpy.asarray(para_y)

        print('las respuestas xz son:\n',resp)
        #print('las respuestas y son:\n',resp_y)
        for i in range(resp.shape[0]):
            resp[i][1][1] = resp_y[i][1][1]

        return resp * nist.ALPHA**4


    def kernel(self, atom1, atom2):
        """This function multiplicates the response by the constants
        in order to get the isotropic J-coupling J between all nuclei in molecule        


        Returns:
            Real: j coupling
        """
        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]
        mol = self.mf.mol
        e11 = self.pp_ssc_4c(atm1lst, atm2lst)
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
        natm = mol.natm
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        jtensor = numpy.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)[0]
        return jtensor