#PBS -S /bin/bash
#PBS -N AMIEGO
#PBS -l select=1:ncpus=11:model=has
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -W group_list=a1607
#PBS -m bae
#PBS -o stdout.out
#PBS -e stderr.out
#PBS -q normal

source ~/.bashrc

cd /u/ktmoore1/OpenMDAO_Beta/AMIEGO-AMD-OpenMDAO

USE_PROC_FILES=1 mpiexec python -u run_parallel_mission_opt.py
