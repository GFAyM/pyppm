from pyscf import gto, scf
from pyscf.gto import Mole
import numpy 
from pyscf import lib
import attr
from pyscf import ao2mo
import numpy as np
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyscf import tools
from pyscf.lib import logger
import sys

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
    mf = attr.ib(default=None, type=scf.hf.RHF, validator=attr.validators.instance_of(scf.hf.RHF))

    def __attrs_post_init__(self):
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mol = self.mf.mol    
        self.nuc_pair = [(i,j) for i in range(self.mol.natm) for j in range(i)]
        self.occidx = numpy.where(self.mo_occ==2)[0]
        self.viridx = numpy.where(self.mo_occ==0)[0]
        self.orbv = self.mo_coeff[:,self.viridx]
        self.orbo = self.mo_coeff[:,self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]

        self.atm1dic, self.atm2dic = uniq_atoms(nuc_pair=self.nuc_pair)

    @property
    def m_matrix_triplet(self):
        r'''A and B matrices for TDDFT response function.

        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        '''
        nao, nmo = self.mo_coeff.shape
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir


        e_ia = lib.direct_sum('a-i->ia', self.mo_energy[self.viridx], self.mo_energy[self.occidx])
        a = numpy.diag(e_ia.ravel()).reshape(self.nocc,self.nvir,self.nocc,self.nvir)
        b = numpy.zeros_like(a)

        eri_mo = ao2mo.general(self.mol, [self.orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(self.nocc,nmo,nmo,nmo)
        a -= numpy.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
        b -= numpy.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])

        m = a + b
        m = m.reshape(self.nocc*self.nvir,self.nocc*self.nvir, order='C')
        
        return m

    def h1_fc_pyscf(self,atmlst):
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mol = self.mol
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



    def _write(stdout, msc3x3, title):
        stdout.write('%s\n' % title)
        stdout.write('mu_x %s\n' % str(msc3x3[0]))
        stdout.write('mu_y %s\n' % str(msc3x3[1]))
        stdout.write('mu_z %s\n' % str(msc3x3[2]))
        stdout.flush()

    @property
    def pp_ssc_fc(self):
        
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.h1_fc_pyscf(sorted(self.atm1dic))
        h2 = self.h1_fc_pyscf(sorted(self.atm2dic))    
        m = self.m_matrix_triplet
        p = np.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        para = []
        for i,j in self.nuc_pair:
            at1 = self.atm1dic[i]
            at2 = self.atm2dic[j]
            e = numpy.einsum('ia,iajb,jb', h1[at1].T, p , h2[at2].T)
            #print(e)
            para.append(e*4)  # *4 for +c.c. and for double occupancy
            
        fc = numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))    
        return fc

    def _atom_gyro_list(self,mol):
        gyro = []
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            if symb in mol.nucprop:
                prop = mol.nucprop[symb]
                mass = prop.get('mass', None)
                gyro.append(get_nuc_g_factor(symb, mass))
            else:
                # Get default isotope
                gyro.append(get_nuc_g_factor(symb))
        return numpy.array(gyro)


    @property
    def kernel(self):
        #log = lib.logger.Logger(sys.stdout, 4)
        #log.verbose = 3
        fc = self.pp_ssc_fc
        nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton ** 2
        iso_ssc = unit * numpy.einsum('kii->k', fc) / 3
        
        #print(iso_ssc)
        
        natm = self.mol.natm
        ktensor = numpy.zeros((natm,natm))
        for k, (i, j) in enumerate(self.nuc_pair):
            ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
        
        gyro = self._atom_gyro_list(self.mol)
        jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
        label = ['%2d %-2s'%(ia, self.mol.atom_symbol(ia)) for ia in range(natm)]
        #log.info( '\nNuclear g factor %s', gyro)
        #log.note(self, 'Spin-spin coupling constant J (Hz)')
        tools.dump_mat.dump_tri(self.mol.stdout, jtensor, label)
        #return ssc

    