#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J webNet_noRes_Tucker24D_CIFAR100
### -- ask for number of cores (default: 1) -- 
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s152576@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o ErrOut/Output_webNet_noRes_Tucker24D_CIFAR100.out 
#BSUB -e ErrOut/Error_webNet_noRes_Tucker24D_CIFAR100.err 

# Load the cuda module
module load cuda/10.2
module load python3/3.8.2
python3 webNet_noRes_Tucker24D_CIFAR100.py


