import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf, lib
from pyscf.dft import numint
from pyscf.data import nist

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import tools
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.scf import _response_functions  # noqa
from pyscf.ao2mo import _ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

def h1_fc_pyscf(atmlst):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mol = mf.mol
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

def _atom_gyro_list(mol):
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

ang=100
mol = gto.M(atom='''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 {}
        '''.format(ang*10), basis='ccpvdz', verbose=0)

mf = scf.RHF(mol).run()

mo_energy = mf.mo_energy
mo_occ = mf.mo_occ
nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
atm1dic, atm2dic = uniq_atoms(nuc_pair=nuc_pair)

h2 = h1_fc_pyscf(sorted(atm2dic.keys()))
h1 = h1_fc_pyscf(sorted(atm1dic.keys()))

#print(h2[0])
#print(numpy.linalg.inv(h2[0]))
#h2_ = numpy.asarray(h2)
#print(h2_.shape)

ppobj = pp(mf)
pol_prop = ppobj.m_matrix_triplet
#print(pol_prop.reshape(9,29,9,29).shape)
pol_prop = numpy.linalg.inv(pol_prop)
princ = pol_prop.reshape(9,29,9,29)

#eai = 1. / lib.direct_sum('a-i->ai', mo_energy[mo_occ==0], mo_energy[mo_occ>0])
#print(eai.shape)
para = []

h1 = h1_fc_pyscf(sorted(atm1dic.keys()))

for i,j in nuc_pair:
    at1 = atm1dic[i]
    at2 = atm2dic[j]
    e = numpy.einsum('ia,iajb,jb', h1[at1].T, princ , h2[at2].T)
    print(e)
    para.append(e*4)  # *4 for +c.c. and for double occupancy



fc = numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))

print(fc)

nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
au2Hz = nist.HARTREE2J / nist.PLANCK
unit = au2Hz * nuc_magneton ** 2
iso_ssc = unit * numpy.einsum('kii->k', fc) / 3
print(iso_ssc)
natm = mol.natm
ktensor = numpy.zeros((natm,natm))
for k, (i, j) in enumerate(nuc_pair):
    ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
#    _write(mol.stdout, fc[k],
#            '\nSSC E11 between %d %s and %d %s'
#            % (i, mol.atom_symbol(i),
#                j, mol.atom_symbol(j)))

gyro = _atom_gyro_list(mol)
jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
#logger.note( 'Reduced spin-spin coupling constant K (Hz)')
#tools.dump_mat.dump_tri(mol.stdout, ktensor, label)
#logger.info( '\nNuclear g factor %s', gyro)
#logger.note( 'Spin-spin coupling constant J (Hz)')
tools.dump_mat.dump_tri(mol.stdout, jtensor, label)


#print(atm2)

