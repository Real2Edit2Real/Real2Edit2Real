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

hf download Real2Edit2Real/Real2Edit2Real-Data --local-dir data/
tar -zxvf data/example_data.tar.gz -C data/
tar -zxvf data/A2D_120s.tar.gz -C data/