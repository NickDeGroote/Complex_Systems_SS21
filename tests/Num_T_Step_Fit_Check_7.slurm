#!/bin/bash
# set the partition where the job will run
#SBATCH --partition=low
# set the number of nodes
#SBATCH --nodes=1
# set the number of CPUs to use per node
#SBATCH --ntasks-per-node=16
# set max wallclock time
#SBATCH --time=100:00:00
# set name of job
#SBATCH --job-name=pickerln_Num_T_Step_Fit_Check_7
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=pickerln@mail.uc.edu
# run the application
python Num_T_Step_Fit_Check_7.py