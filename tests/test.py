from pyscf import scf, gto 

from src.ssc_pol_prop import Prop_pol



HF_mol = gto.M(atom="""H 0 0 0; F 1 0 0""", basis="cc-pvdz", unit="angstrom")
mf = scf.RHF(HF_mol)
mf.kernel()
#pert_fcsd = Prop_pol(mf).pert_fcsd([0])
#print((pert_fcsd[0][1]*pert_fcsd[0][1]).sum())
#print(Prop_pol(mf).pp_fc([0], [1])[0][0][0])
print(Prop_pol(mf)._get_integrals_fcsd(1).sum())