#!/bin/bash
# Usage: ./setup.sh [--force-pip] [--force-models] [--force]

FORCE_PIP=false
FORCE_MODELS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-pip)
            FORCE_PIP=true
            shift
            ;;
        --force-models)
            FORCE_MODELS=true
            shift
            ;;
        --force)
            FORCE_PIP=true
            FORCE_MODELS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force-pip] [--force-models] [--force]"
            exit 1
            ;;
    esac
done

# Pip packages
if [ "$FORCE_PIP" = true ]; then
    pip install -r requirements.txt --force-reinstall
else
    pip install -r requirements.txt
fi

# SDXL Lightning model
mkdir -p ckpt
if [ ! -f "ckpt/sdxl_lightning_4step_unet.safetensors" ] || [ "$FORCE_MODELS" = true ]; then
    wget https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_unet.safetensors -O ckpt/sdxl_lightning_4step_unet.safetensors
fi

# Face landmarks model
mkdir -p models
if [ ! -f "models/shape_predictor_5_face_landmarks.dat" ] || [ "$FORCE_MODELS" = true ]; then
    rm -f models/shape_predictor_5_face_landmarks.dat
    wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 -O models/shape_predictor_5_face_landmarks.dat.bz2
    bzip2 -d models/shape_predictor_5_face_landmarks.dat.bz2
fi

# FairFace model
if [ ! -f "models/res34_fair_align_multi_7_20190809.pt" ] || [ "$FORCE_MODELS" = true ]; then
    gdown "1fUJSLseDpgilArB_YKep9PnsR7QrPW5I" -O models/res34_fair_align_multi_7_20190809.pt
fi

# Verify
ls -lh models/