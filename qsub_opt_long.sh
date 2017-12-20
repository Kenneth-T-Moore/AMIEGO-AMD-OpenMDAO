#PBS -S /bin/bash
#PBS -N AMIEGO
#PBS -l select=1:ncpus=1:mpiprocs=1:model=bro+5:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -W group_list=a1607
#PBS -m bae
#PBS -o stdout.out
#PBS -e stderr.out
#PBS -q long

source ~/.bashrc

cd /u/ktmoore1/OpenMDAO_Beta/AMIEGO-AMD-OpenMDAO

mpiexec python -u run_amd_opt.py
