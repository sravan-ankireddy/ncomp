#!/bin/bash

# # Set the rate inside the script
# RATE="0.1250"
# INPUT_IMAGE="/work/09004/sravana/ls6/ncomp/perco/res/eval/kodim13.png"
# OUTPUT_LATENTS="/work/09004/sravana/ls6/ncomp/perco/res/outputs/latents/kodim13.pkl"
# MODEL_PATH="/work/09004/sravana/ls6/ncomp/perco/src/models_chk/$RATE/cmvl_2024_full_train"

# # compress
# python /work/09004/sravana/ls6/ncomp/perco/src/compression_utils.py -V \
#   --model_path "$MODEL_PATH" \
#   compress "$INPUT_IMAGE" \
#   "$OUTPUT_LATENTS"


#!/bin/bash

# Set the rate inside the script
RATE="0.0019"
DATASET="MSCOCO30k"
MODEL_PATH="/work/09004/sravana/ls6/ncomp/perco/src/models_chk/$RATE/cmvl_2024_full_train"

# IN_DIR="/work/09004/sravana/ls6/ncomp/perco/src/data/$DATASET/"
IN_DIR="/scratch/09004/sravana/MSCOCO/$DATASET/"

OUT_DIR="/work/09004/sravana/ls6/ncomp/perco/res/outputs/$DATASET/$RATE/"

# Evaluate dataset
python /work/09004/sravana/ls6/ncomp/perco/src/compression_utils.py --model_path "$MODEL_PATH" \
    evaluate_ds --mode 0 --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
