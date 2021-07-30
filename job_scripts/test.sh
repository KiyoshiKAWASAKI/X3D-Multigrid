#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N test_hmdb_with_feedback

# Required modules
module load conda
conda init bash
source activate new_x3d

python test_x3d_ta2_cls.py