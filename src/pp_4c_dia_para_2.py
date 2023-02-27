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

def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

@attr.s
class Prop_pol:
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
        occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
        #viridx = numpy.where(mo_occ[:n2c] == 0)[0]
        viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
        self.orbv = mo_coeff[:,viridx]
        self.orbo = mo_coeff[:,occidx]
        
        nvir = self.orbv.shape[1]
        nocc = self.orbo.shape[1]
        nmo = nocc + nvir
        c1 = .5 / lib.param.LIGHT_SPEED
        mo = numpy.hstack((self.orbo, self.orbv))
        moL = numpy.asarray(mo[:n2c], order='F')
        moS = numpy.asarray(mo[n2c:], order='F')* c1

        
        orboL = moL[:,:nocc]
        orboS = moS[:,:nocc]
        e_ia = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])
        #a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
        a = numpy.zeros((nocc*nvir,nocc*nvir), dtype='complex128')
        numpy.fill_diagonal(a,e_ia.ravel())
        #print(numpy.trace(a))
        a = a.reshape(nvir,nocc,nvir,nocc)

        eri_mo = ao2mo.kernel(mol, [moL, moL, moL, moL], intor='int2e_spinor')
        eri_mo+= ao2mo.kernel(mol, [moS, moS, moS, moS], intor='int2e_spsp1spsp2_spinor')
        eri_mo+= ao2mo.kernel(mol, [moS, moS, moL, moL], intor='int2e_spsp1_spinor')
        eri_mo+= ao2mo.kernel(mol, [moS, moS, moL, moL], intor='int2e_spsp1_spinor').T
        #print('done')
        eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
        
        a +=  numpy.einsum('aijb->aibj', eri_mo[nocc:,:nocc,:nocc,nocc:])
        a -=  numpy.einsum('abij->aibj', eri_mo[nocc:,nocc:,:nocc,:nocc])
        b = numpy.einsum('jaib->aibj', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b -=  numpy.einsum('jbia->aibj', eri_mo[:nocc,nocc:,:nocc,nocc:])

        m = a + b

        a = numpy.zeros((nocc*nvir,nocc*nvir), dtype='complex128')
        numpy.fill_diagonal(a,e_ia.ravel())
        a = a.reshape(nvir,nocc,nvir,nocc)
        a +=  numpy.einsum('aijb->aibj', eri_mo[nocc:,:nocc,:nocc,nocc:])
        a -=  numpy.einsum('abij->aibj', eri_mo[nocc:,nocc:,:nocc,:nocc])
        b = numpy.einsum('jaib->aibj', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b -=  numpy.einsum('jbia->aibj', eri_mo[:nocc,nocc:,:nocc,nocc:])
        
        m_y = a-b

        m = m.reshape(nocc*nvir,nocc*nvir)
        m_y = m_y.reshape(nocc*nvir,nocc*nvir)
        
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
        #orbo = mo_coeff[:,mo_occ==1]
        #orbv = mo_coeff[:,mo_occ==0]
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
                h1.append(self.orbv.conj().T.dot(h01).dot(self.orbo))
        return h1

    def pp_ssc_4c(self):
        """In this Function generate de Response << ; >>, i.e, multiplicate the perturbators centered
        in all nuclei with the principal propagator.


        Returns:
            numpy.array: J response
        """
        mol = self.mf.mol
        #mo_coeff = self.mf.mo_coeff
        #mo_occ = self.mf.mo_occ
        #orbo = mo_coeff[:,mo_occ==1]
        #orbv = mo_coeff[:,mo_occ==0]
        m_xz, m_y = self.M
        #print(numpy.isnan(m_xz).any())
        
        p_xz = -numpy.linalg.inv(m_xz)
        p_y = -numpy.linalg.inv(m_y)
        #print(numpy.trace(p_xz))

        nvir = self.orbv.shape[1]
        nocc = self.orbo.shape[1]

        p_xz = p_xz.reshape(nvir,nocc,nvir,nocc)

        p_y = p_y.reshape(nvir,nocc,nvir,nocc)

        nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        atm1lst = sorted(set([i for i,j in nuc_pair]))
        atm2lst = sorted(set([j for i,j in nuc_pair]))
        
        atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
        atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])

        h1 = numpy.asarray(self.make_perturbator(atm1lst)).reshape(len(atm1lst),3,nvir,nocc)        
        
        h2 = numpy.asarray(self.make_perturbator(atm2lst)).reshape(len(atm2lst),3,nvir,nocc)
        #print(h1[0][0].sum())
        

        para = []
        para_y = []

        for i,j in nuc_pair:
            e = numpy.einsum('xai,aibj,ybj->xy', h1[atm1dic[i]], p_xz.conj(), h2[atm2dic[j]].conj()) * 2
            para.append(e.real)
            e_y = numpy.einsum('xai,aibj,ybj->xy', h1[atm1dic[i]], p_y.conj(), h2[atm2dic[j]].conj()) * 2
            para_y.append(e_y.real)
            
        resp = numpy.asarray(para)
        resp_y = numpy.asarray(para_y)

        for i in range(resp.shape[0]):
            resp[i][1][1] = resp_y[i][1][1]

        print(resp * nist.ALPHA**4)

        return resp * nist.ALPHA**4


    def kernel(self):
        """This function multiplicates the response by the constants
        in order to get the isotropic J-coupling J between all nuclei in molecule        


        Returns:
            Real: j coupling
        """
        
        mol = self.mf.mol
        e11 = self.pp_ssc_4c()
       
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
        natm = mol.natm
        ktensor = numpy.zeros((natm,natm))
        for k, (i, j) in enumerate(nuc_pair):
            ktensor[i,j] = ktensor[j,i] = iso_ssc[k]

        gyro = [get_nuc_g_factor(mol.atom_symbol(ia)) for ia in range(natm)]
        jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
        label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
        tools.dump_mat.dump_tri(mol.stdout, jtensor, label)
        
        return jtensor






