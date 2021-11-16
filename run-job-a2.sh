#!/bin/bash

##First load all related modules.  
#You can put the below two lines in a batch file. 
#But remember the modules might get unloaded so check that you have loaded modules frequently. 
module load anaconda3/5.2.0
module load gcc/7.3.0
module load cmake 

#Generate the data
rm -f data.pickle # This ensures that you are not using the profiling data from the last run.
make pgm_creator
./pgm_creator.out

#Compile your code
make clean # The first time you run this script, you might get an error/warning which is normal, ignore it!
make main

#Schedule your jobs with sbatch
sbatch job-a2.sh

