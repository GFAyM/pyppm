from pyscf import gto, scf, dft
from pyscf.prop import ssc
from pyscf.dft import numint
import numpy
mol = gto.M(atom='''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 100 ''',
    basis='ccpvdz')

mf = scf.RHF(mol).run()
sc = ssc.RHF(mf)

def _uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

nuc_pa = sc.nuc_pair

atm1dic, atm2dic = _uniq_atoms(nuc_pa)

def make_h1_fc(mol, mo_coeff, mo_occ, atmlst):
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


from pyscf.lib import logger
from pyscf import lib
from pyscf.ao2mo import _ao2mo

def solve_mo1_fc(sscobj, h1):
    cput1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(sscobj.stdout, sscobj.verbose)
    mo_energy = sscobj._scf.mo_energy
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    nset = len(h1)
    eai = 1. / lib.direct_sum('a-i->ai', mo_energy[mo_occ==0], mo_energy[mo_occ>0])
    mo1 = numpy.asarray(h1) * -eai
    if not sscobj.cphf:
        return mo1

    orbo = mo_coeff[:,mo_occ> 0]
    orbv = mo_coeff[:,mo_occ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    nmo = nocc + nvir

    vresp = sscobj._scf.gen_response(singlet=False, hermi=1)
    mo_v_o = numpy.asarray(numpy.hstack((orbv,orbo)), order='F')
    def vind(mo1):
        dm1 = _dm1_mo2ao(mo1.reshape(nset,nvir,nocc), orbv, orbo*2)  # *2 for double occupancy
        dm1 = dm1 + dm1.transpose(0,2,1)
        v1 = vresp(dm1)
        v1 = _ao2mo.nr_e2(v1, mo_v_o, (0,nvir,nvir,nmo)).reshape(nset,nvir,nocc)
        v1 *= eai
        return v1.ravel()

    mo1 = lib.krylov(vind, mo1.ravel(), tol=sscobj.conv_tol,
                     max_cycle=sscobj.max_cycle_cphf, verbose=log)
    log.timer('solving FC CPHF eqn', *cput1)
    return mo1.reshape(nset,nvir,nocc)

def _dm1_mo2ao(dm1, ket, bra):
    nao, nket = ket.shape
    nbra = bra.shape[1]
    nset = len(dm1)
    dm1 = lib.ddot(ket, dm1.transpose(1,0,2).reshape(nket,nset*nbra))
    dm1 = dm1.reshape(nao,nset,nbra).transpose(1,0,2).reshape(nset*nao,nbra)
    return lib.ddot(dm1, bra.T).reshape(nset,nao,nao)