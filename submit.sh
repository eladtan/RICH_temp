#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=staging
#SBATCH --time=10:18:00
#SBATCH --output=output_%j
#SBATCH --error=error_%j
 
# Load modules for MPI and other parallel libraries
ml restore
 
# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
 
# Execute the program in parallel on ntasks cores
 
#srun ./rich 

 #/home/esternberg/rclone-v1.61.1-linux-amd64/rclone copy drive:/SimData/Snellius/R1M1BH10000beta1S60Compton/snap_full_378.h5 /scratch-shared/esternberg/R1M1BH10000beta1S60Compton --progress --transfers 12

/home/esternberg/rclone-v1.61.1-linux-amd64/rclone copy /scratch-shared/esternberg/R2.4M3BH10000beta2S60n3Compton drive:/SimData/Snellius/R2.4M3BH10000beta2S60n3Compton  --progress --transfers 12
# /home/esternberg/rclone-v1.61.1-linux-amd64/rclone copy /scratch-shared/esternberg/R2.4M3BH100000beta2S60n3Compton drive:/SimData/Snellius/R2.4M3BH100000beta2S60n3Compton  --progress --transfers 12
# /home/esternberg/rclone-v1.61.1-linux-amd64/rclone copy /scratch-shared/esternberg/R1M1BH10000beta2S60n3Compton drive:/SimData/Snellius/R1M1BH10000beta2S60n3Compton  --progress --transfers 12
# /home/esternberg/rclone-v1.61.1-linux-amd64/rclone copy /scratch-shared/esternberg/R1M1BH100000beta2S60n3Compton drive:/SimData/Snellius/R1M1BH100000beta2S60n3Compton  --progress --transfers 12
# /home/esternberg/rclone-v1.61.1-linux-amd64/rclone copy /scratch-shared/esternberg/R0.47M0.5BH100000beta1S60n1.5Compton drive:/SimData/Snellius/R0.47M0.5BH100000beta1S60n1.5Compton  --progress --transfers 12
