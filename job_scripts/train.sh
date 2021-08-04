#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N train_hmdb_0802

# Required modules
module load conda
conda init bash
source activate new_x3d

python train_x3d_ta2_cls.py