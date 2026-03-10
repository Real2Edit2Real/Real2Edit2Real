#!/bin/bash

# ==============================================
# [Optional] Use China HuggingFace mirror?
# ==============================================
USE_HF_MIRROR=${USE_HF_MIRROR:-false}

if [ "$USE_HF_MIRROR" = true ] || [ "$USE_HF_MIRROR" = "1" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "Using HuggingFace mirror at $HF_ENDPOINT"
else
    echo "Using default HuggingFace endpoint"
fi

# ----------------------------------------------------
# 1. Download Real2Edit2Real Pretrained Weights
hf download Real2Edit2Real/Real2Edit2Real --local-dir checkpoints/

# ----------------------------------------------------
# 2. Download bert-base-uncased for GroundingDINO
hf download google-bert/bert-base-uncased --local-dir checkpoints/bert-base-uncased

# ----------------------------------------------------
# 3. Download groundingdino_swinb_cogcoor.pth
# Check if the target file exists.
TARGET_FILE_1="checkpoints/groundingdino_swinb_cogcoor.pth"
if [ ! -f "$TARGET_FILE_1" ]; then
    echo "File $TARGET_FILE_1 not found. Starting download..."
    aria2c -s16 -x16 -k1M https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth -o "$TARGET_FILE_1"
else
    echo "File $TARGET_FILE_1 already exists. Skipping download."
fi

# ----------------------------------------------------
# 4. Download sam2_hiera_large.pth
# Check if the target file exists.
TARGET_FILE_1="checkpoints/sam2_hiera_large.pt"
if [ ! -f "$TARGET_FILE_1" ]; then
    echo "File $TARGET_FILE_1 not found. Starting download..."
    aria2c -s16 -x16 -k1M https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -o "$TARGET_FILE_1"
else
    echo "File $TARGET_FILE_1 already exists. Skipping download."
fi

# ----------------------------------------------------
# 5. Download Cosmos-Predict2 for Video Generation
hf download nvidia/Cosmos-Predict2-2B-Video2World --local-dir checkpoints/Cosmos-Predict2-2B-Video2World

