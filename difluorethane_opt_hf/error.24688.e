/var/spool/slurm/job24688/slurm_script: línea 22: cd: /home/danielba/Pyscf-2021/module/difluorethane: No existe el fichero o el directorio
Traceback (most recent call last):
  File "entanglement_difluorethane_3exc.py", line 67, in <module>
    mol, mo_coeff = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang}_Cholesky_PM.molden").extraer_coeff
  File "/home/danielba/pyPPE/src/help_functions.py", line 13, in extraer_coeff
    self.mol, mo_energy, self.mo_coeff, mo_occ, irrep_labels, spins =  tools.molden.load(self.molden_file)
  File "/cluster/software/PySCF/2.0.0a-foss-2020b-Python-3.8.6/pyscf/tools/molden.py", line 322, in load
    with open(moldenfile, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'difluorethane_cc-pvdz_1_Cholesky_PM.molden'
