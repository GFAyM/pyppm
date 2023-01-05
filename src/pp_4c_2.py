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
    print("total memory: %.1f MiB" % current_memory()[0])

        
    
    def A1(self):
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
        mo_energy = self.mf.mo_energy
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
        orbvL = moL[:,nocc:]
        orbvS = moS[:,nocc:] 

        e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
        a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
        
        a1 = ao2mo.general(mol, [orboL, orbvL, orbvL, orboL], intor='int2e_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a1 += ao2mo.general(mol, [orboS, orbvS, orbvS, orboS], intor='int2e_spsp1spsp2_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a1 += ao2mo.general(mol, [orboS, orbvS, orbvL, orboL], intor='int2e_spsp1_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a1 += ao2mo.general(mol, [orbvS, orboS, orboL, orbvL], intor='int2e_spsp1_spinor').T
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a1 = a1.reshape(nocc,nvir,nvir,nocc)

        a = a + lib.einsum('iabj->iajb', a1)

        return a

    def A2(self):
        """
        """
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
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

        a2 = ao2mo.general(mol, [orboL, orboL, orbvL, orbvL], intor='int2e_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a2 += ao2mo.general(mol, [orboS, orboS, orbvS, orbvS], intor='int2e_spsp1spsp2_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a2+= ao2mo.general(mol, [orboS, orboS, orbvL, orbvL], intor='int2e_spsp1_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a2+= ao2mo.general(mol, [orbvS, orbvS, orboL, orboL], intor='int2e_spsp1_spinor').T
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        a2 = a2.reshape(nocc,nocc,nvir,nvir)
        a2 = lib.einsum('ijba->iajb', a2)
        return a2

    def B(self):
        
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
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

        
        b_ = ao2mo.general(mol, [orboL, orbvL, orboL, orbvL], intor='int2e_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        b_ += ao2mo.general(mol, [orboS, orbvS, orboS, orbvS], intor='int2e_spsp1spsp2_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        b_+= ao2mo.general(mol, [orboS, orbvS, orboL, orbvL], intor='int2e_spsp1_spinor')
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        b_+= ao2mo.general(mol, [orboS, orbvS, orboL, orbvL], intor='int2e_spsp1_spinor').T
        print("integral","total memory: %.1f MiB" % current_memory()[0])
        b_ = b_.reshape(nocc,nvir,nocc,nvir)
        b_ =  lib.einsum('iajb->iajb', b_)
        b_ -= lib.einsum('jaib->iajb', b_)
        return b_    
        
    @property
    def M(self):
        
        a1 = self.A1() -self.A2()
        print("a1 total memory: %.1f MiB" % current_memory()[0])
        
        b = self.B()
        print("b total memory: %.1f MiB" % current_memory()[0])
        
        m_xz = a1  + b
        print("total memory: %.1f MiB" % current_memory()[0])
        
        m_y = a1 - b
        print("total memory: %.1f MiB" % current_memory()[0])
        
        return m_xz, m_y


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
        orbo = mo_coeff[:,mo_occ> 0]
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

    def pp_ssc_4c(self):
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

        nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
        atm1lst = sorted(set([i for i,j in nuc_pair]))
        atm2lst = sorted(set([j for i,j in nuc_pair]))
        
        atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
        atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])

        h1 = numpy.asarray(self.make_perturbator(atm1lst)).reshape(len(atm1lst),3,nvir,nocc)        
        print('h1',"total memory: %.1f MiB" % current_memory()[0])
        
        h2 = numpy.asarray(self.make_perturbator(atm2lst)).reshape(len(atm2lst),3,nvir,nocc)
        print('h2',"total memory: %.1f MiB" % current_memory()[0])
        
        print('before call p',"total memory: %.1f MiB" % current_memory()[0])

        m_xz, m_y = self.M
        m_xz = m_xz.reshape(nocc*nvir,nocc*nvir, order='C')
        m_y = m_y.reshape(nocc*nvir,nocc*nvir, order='C')
        print('after call p',"total memory: %.1f MiB" % current_memory()[0])
        p_xz = -numpy.linalg.inv(m_xz)        
        p_xz = p_xz.reshape(nocc,nvir,nocc,nvir)
        p_y = -numpy.linalg.inv(m_y)
        p_y = p_y.reshape(nocc,nvir,nocc,nvir)

        print('after invert the matrices',"total memory: %.1f MiB" % current_memory()[0])

        



        para = []
        para_y = []
        print('Before initialize pp response',"total memory: %.1f MiB" % current_memory()[0])
        for i,j in nuc_pair:
            e = lib.einsum('xai,iajb,ybj->xy', h1[atm1dic[i]], p_xz.conj(), h2[atm2dic[j]].conj()) * 2
            para.append(e.real)
            e_y = lib.einsum('xai,iajb,ybj->xy', h1[atm1dic[i]], p_y.conj(), h2[atm2dic[j]].conj()) * 2
            para_y.append(e_y.real)
            
        resp = numpy.asarray(para)
        resp_y = numpy.asarray(para_y)
        for i in range(e.shape[0]):
            resp[i][1][1] = resp_y[i][1][1]
        print('ell modulo pp funciona')
        print('after finalize pp response',"total memory: %.1f MiB" % current_memory()[0])
        #print(numpy.asarray(para) * nist.ALPHA**4)

        return resp * nist.ALPHA**4

    def pp_ssc_4c_select(self,atm1lst,atm2lst):
        """
        In this Function generate de Response << ; >>, i.e, multiplicate the perturbators centered in
        the nuclei of election, with the principal propagator.

        Args:
            atm1lst (list): Nuclei A
            atm2lst (list): Nuclei B

        Returns:
            numpy.array: << ; >>
        """
        nocc = self.nocc
        nvir = self.nvir
        h1 = numpy.asarray(self.make_perturbator(atm1lst)).reshape(len(atm1lst),3,self.nvir,self.nocc)

        h2 = numpy.asarray(self.make_perturbator(atm2lst)).reshape(len(atm2lst),3,self.nvir,self.nocc)
        m = self.M

        p = -numpy.linalg.inv(m)
        p = p.reshape(nocc,nvir,nocc,nvir)
        

        para = []
        
        #e1 = lib.einsum('iajb,ybj->yia',  p, h2[0])
            #e1 = p * h2[atm2dic[j]]
            #print(e1[0])
        #print(h1[0].shape)
        #print(e1.shape)
        #e = lib.einsum('xai,yia->yx', h1[0], e1.conj()) * 2
        #print(h1[0][0].shape)
        e = lib.einsum('ai,iajb,bj->', h1[0][0], p.conj(), h2[0][1].conj()) * 2
        print(numpy.asarray(e.real)*nist.ALPHA**4)
        e = lib.einsum('ai,iajb,bj->', h1[0][2], p.conj(), h2[0][2].conj()) * 2
        print(numpy.asarray(e.real)*nist.ALPHA**4)
        e = lib.einsum('ai,iajb,bj->', h1[0][2], p.conj(), h2[0][1].conj()) * 2
        print(numpy.asarray(e.real)*nist.ALPHA**4)
        e = lib.einsum('xai,iajb,ybj->xy', h1[0], p.conj(), h2[0].conj()) * 2
        para.append(e.real)
        return numpy.asarray(para) * nist.ALPHA**4


    

    def kernel_select(self,atom1,atom2):
        """This function multiplicates the response by the constants
        in order to get the isotropic J-coupling J(A,B) between the two nuclei        

        Args:
            atom1 (list): Nuclei A
            atom2 (list): Nuclei B

        Returns:
            Real: j coupling
        """
        mol = self.mol

        e11 = self.pp_ssc_4c_select(atom1,atom2)
        print(e11)
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * lib.einsum('kii->k', e11) / 3
        gyro1 = [get_nuc_g_factor(mol.atom_symbol(atom1[0]))]
        gyro2 = [get_nuc_g_factor(mol.atom_symbol(atom2[0]))]
        jtensor = lib.einsum('i,i,j->ij', iso_ssc, gyro1, gyro2)
        return jtensor

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
        iso_ssc = au2Hz * nuc_mag ** 2 * lib.einsum('kii->k', e11) / 3
        #print(iso_ssc)
        natm = mol.natm
        ktensor = numpy.zeros((natm,natm))
        for k, (i, j) in enumerate(nuc_pair):
            ktensor[i,j] = ktensor[j,i] = iso_ssc[k]

        gyro = [get_nuc_g_factor(mol.atom_symbol(ia)) for ia in range(natm)]
        jtensor = lib.einsum('ij,i,j->ij', ktensor, gyro, gyro)
        label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
#        print(lib.param.LIGHT_SPEED)
        tools.dump_mat.dump_tri(mol.stdout, jtensor, label)
        
        return jtensor




