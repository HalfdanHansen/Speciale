#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J Tucker2
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
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
#BSUB -o Output/0804/Output_CIFAR10_no_pass_net_Tucker2_ranks.out 
#BSUB -e Output/0804/Error_CIFAR10_no_pass_net_Tucker2_ranks.err 

# Load the cuda module
module load cuda/10.2
module load python3/3.8.2
cd Documents/
python3 0704_CIFAR10_no_pass_net_Tucker2_ranks.py


