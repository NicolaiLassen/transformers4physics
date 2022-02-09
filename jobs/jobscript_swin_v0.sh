#!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J con
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -R "span[hosts=1]"
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 module load python3/3.6.2
 pip3 install --user -r requirements.txt
 echo "Traning..."
 python3 pre-training_train.py -l "con"