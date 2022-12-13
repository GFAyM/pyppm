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

def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

@attr.s
class Prop_pol:
    """_summary_
    """
    mf = attr.ib(default=None, type=scf.dhf.DHF, validator=attr.validators.instance_of(scf.dhf.DHF))

    def __attrs_post_init__(self):
        self.mol = self.mf.mol
        self.mo_coeff = self.mf.mo_coeff
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.n4c, self.nmo = self.mo_coeff.shape
        
        self.n2c = self.nmo // 2
        self.nuc_pair = [(i,j) for i in range(self.mol.natm) for j in range(i)]
        #self.occidx = self.n2c + numpy.where(self.mo_occ[self.n2c:] == 1)[0]
        #self.viridx = self.n2c + numpy.where(self.mo_occ[self.n2c:] == 0)[0]
        self.occidx = numpy.where(self.mo_occ > 0)[0]
        self.viridx = numpy.where(self.mo_occ == 0)[0]
        
        self.orbv = (self.mo_coeff[:,self.viridx])
        self.orbo = (self.mo_coeff[:,self.occidx])
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.nmo = self.nocc + self.nvir
        self.mo = numpy.hstack((self.orbo, self.orbv))
        self.c1 = .5 / lib.param.LIGHT_SPEED
        self.moL = numpy.asarray(self.mo[:self.n2c], order='F')
        self.moS = numpy.asarray(self.mo[self.n2c:], order='F')* self.c1
        self.orboL = self.moL[:,:self.nocc]
        self.orboS = self.moS[:,:self.nocc]
        #self.atm1dic, self.atm2dic = uniq_atoms(nuc_pair=self.nuc_pair)
    @property
    def M(self):
        r'''A and B matrices for TDDFT response function.

        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        '''
        mol = self.mol
        nmo = self.nmo
        mo_energy = self.mo_energy
        viridx = self.viridx
        occidx = self.occidx
        nocc = self.nocc
        nvir = self.nvir
        moL = self.moL 
        moS = self.moS
        orboL = self.orboL
        orboS = self.orboS
        e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
        a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir).astype(complex)
        
        b = numpy.zeros_like(a, dtype=numpy.complex128)
        
        eri_mo = ao2mo.general(mol, [orboL, moL, moL, moL], intor='int2e_spinor')
        eri_mo+= ao2mo.general(mol, [orboS, moS, moS, moS], intor='int2e_spsp1spsp2_spinor')
        eri_mo+= ao2mo.general(mol, [orboS, moS, moL, moL], intor='int2e_spsp1_spinor')
        eri_mo+= ao2mo.general(mol, [moS, moS, orboL, moL], intor='int2e_spsp1_spinor').T
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        
        a = a + numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
        a = a - numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) 
        b = b + numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b = b - numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) 
        m = a+b

        m = m.reshape(nocc*nvir,nocc*nvir)
        return m

    @property
    def M_myway(self):
        mol = self.mol
        mo = self.mo
        nocc = self.nocc
        nvir = self.nvir
        n2c = self.n2c
        c = self.c1 
        mo_l = mo[:n2c,:]
        mo_s = mo[n2c:,:] * (.5/c)
        Lo = mo_l[:,:nocc]
        So = mo_s[:,:nocc]
        Lv = mo_l[:,nocc:]
        Sv = mo_s[:,nocc:]
        mo_energy = self.mo_energy
        viridx = self.viridx 
        occidx = self.occidx

        eri_a1 = ao2mo.general(mol,  (Lv, Lo, Lo, Lv), intor='int2e_spinor')
        eri_a1 += ao2mo.general(mol, (Sv, So, So, Sv), intor='int2e_spsp1spsp2_spinor')
        eri_a1 += ao2mo.general(mol, (Sv, So, Lo, Lv), intor='int2e_spsp1_spinor'     )
        eri_a1 += ao2mo.general(mol, (Lv, Lo, So, Sv), intor='int2e_spsp2_spinor'     )
        eri_a1 = eri_a1.reshape(nvir,nocc,nocc,nvir)

        eri_a2 = ao2mo.general(mol,  (Lv, Lv, Lo, Lo), intor='int2e_spinor')
        eri_a2 += ao2mo.general(mol, (Sv, Sv, So, So), intor='int2e_spsp1spsp2_spinor')
        eri_a2 += ao2mo.general(mol, (Sv, Sv, Lo, Lo), intor='int2e_spsp1_spinor'     )
        eri_a2 += ao2mo.general(mol, (Lv, Lv, So, So), intor='int2e_spsp2_spinor'     )
        eri_a2 = eri_a2.reshape(nvir,nvir,nocc,nocc)

        e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
        a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
        eri_a1 = numpy.einsum('aijb->iajb', eri_a1)
        eri_a2 = numpy.einsum('abji->iajb', eri_a2)

        m = a + eri_a1 - eri_a2
        
        return m.reshape(nocc*nvir,nocc*nvir)



    def make_h1(self, atmlst):
        h1 = []
        for ia in atmlst:
            self.mol.set_rinv_origin(self.mol.atom_coord(ia))
            a01int = self.mol.intor('int1e_sa01sp_spinor', 3)
            h01 = numpy.zeros((self.n4c,self.n4c), numpy.complex128)
            for k in range(3):
                h01[:self.n2c,self.n2c:] = .5 * a01int[k]
                h01[self.n2c:,:self.n2c] = .5 * a01int[k].conj().T
                h1.append(self.orbv.conj().T.dot(h01).dot(self.orbo))
        return h1


    def pp_ssc_4c(self):
        nocc = self.nocc
        nvir = self.nvir
        atm1lst = sorted(set([i for i,j in self.nuc_pair]))
        atm2lst = sorted(set([j for i,j in self.nuc_pair]))
        atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
        atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
        
        m = self.M
        
        p = -numpy.linalg.inv(m)

        p = p.reshape(nocc,nvir,nocc,nvir)
        
        h1 = numpy.asarray(self.make_h1(atm1lst)).reshape(len(atm1lst),3,self.nvir,self.nocc)
        h2 = numpy.asarray(self.make_h1(atm2lst)).reshape(len(atm1lst),3,self.nvir,self.nocc)
        para = []
        for i,j in self.nuc_pair:
            e = numpy.einsum('xai,iajb,ybj->xy', h1[atm1dic[i]], p, h2[atm2dic[j]].conj()) * 2
            para.append(e.real)
        return numpy.asarray(para) * nist.ALPHA**4

    def pp_ssc_4c_select(self,atm1lst,atm2lst):
        nocc = self.nocc
        nvir = self.nvir

        #m = self.M
        m = self.M_myway

        p = -numpy.linalg.inv(m)
        #p = -scipy.linalg.inv(m)
        p = p.reshape(nocc,nvir,nocc,nvir)
        #p = numpy.real(p)
        h1 = numpy.asarray(self.make_h1(atm1lst)).reshape(len(atm1lst),3,self.nvir,self.nocc)
        h2 = numpy.asarray(self.make_h1(atm2lst)).reshape(len(atm1lst),3,self.nvir,self.nocc)
        para = []
        
        print(h1[0][2][5,6])
        print(p[5,6,7,8])
        print(h2[0][2][7,8].conj())
        e = numpy.einsum('xai,iajb,ybj->xy', h1[0], p, h2[0].conj()) * 2
        para.append(e)
        return numpy.asarray(para) * nist.ALPHA**4


    

    def kernel_select(self,atom1,atom2):
        mol = self.mol

        e11 = self.pp_ssc_4c_select(atom1,atom2)
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
        gyro1 = [get_nuc_g_factor(mol.atom_symbol(atom1[0]))]
        gyro2 = [get_nuc_g_factor(mol.atom_symbol(atom2[0]))]
        jtensor = numpy.einsum('i,i,j->ij', iso_ssc, gyro1, gyro2)
        return jtensor

    def kernel(self):
        mol = self.mol

        e11 = self.pp_ssc_4c()
        #e11 = self.ssc_4c_pyscf()
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
        natm = mol.natm
        ktensor = numpy.zeros((natm,natm))
        for k, (i, j) in enumerate(self.nuc_pair):
            ktensor[i,j] = ktensor[j,i] = iso_ssc[k]

        gyro = [get_nuc_g_factor(mol.atom_symbol(ia)) for ia in range(natm)]
        jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
        label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
        tools.dump_mat.dump_tri(self.mol.stdout, jtensor, label)
        
        return jtensor
