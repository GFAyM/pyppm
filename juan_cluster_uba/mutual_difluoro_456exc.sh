#!/bin/bash
#SBATCH --partition=Q2
#SBATCH -o info.%j.o  #Nombre del archivo de salida
#SBATCH -e error.%j.e  #STDERR
#SBATCH -J Pyscf
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=1800mb

#--------------------------------------------------------------------------------------------------------------------------------------------
#SECCION DEFINICIONES
source $HOME/.bashrc
#module data/h5py/3.1.0-foss-2020b chem/qcint/4.0.6-foss-2020b-Python-3.8.6 chem/libxc/5.1.3-GCC-10.2.0 chem/XCFun/2.1.1-GCCcore-10.2.0
module load devel/CMake/3.18.4-GCCcore-10.2.0 data/h5py/3.1.0-foss-2020b chem/qcint/4.0.6-foss-2020b-Python-3.8.6 chem/libxc/5.1.3-GCC-10.2.0 chem/XCFun/2.1.1-GCCcore-10.2.0
#module load lib/pandas/1.1.2-foss-2020a-Python-3.8.2
module load vis/plotly.py/4.14.3-GCCcore-10.2.0
module load chem/PySCF/2.0.0a-foss-2020b-Python-3.8.6
export PYTHONPATH=$HOME/pyPPE/src:$PYTHONPATH
workdir="/home/danielba/Pyscf-2021/module/difluorethane_opt_hf"
scratch="--scratch=/scratch/$USER"
#--------------------------------------------------------------------------------------------------------------------------------------------

cd $workdir

#cp ent_spin_cruz_4exc_AB_terminus.py ent_spin_cruz_4exc_AB_terminus_enuso.py

#sed -i "s/CPUSPERTASK/$SLURM_CPUS_PER_TASK/g" ent_spin_cruz_4exc_AB_terminus_enuso.py

python entropy_iajb_difluorethane_3456exc.py

#rm ent_spin_cruz_4exc_AB_terminus_enuso.py

exit

