#!/bin/bash

# Prompt optimization parameters
t_lo=0.0
N=10
lr=0.01

# "original-csv": "sample_original_prompts.csv",
# "prompt-col": "prompt",
# "enhanced-csv": None,
# "enhanced-col": "enhanced_prompt",

# Create logs directory if it doesn't exist
# -p flag: create parent directories as needed, no error if already exists
mkdir -p logs
mkdir -p logs/csv
mkdir -p logs/csv/main

# Get timestamp for log file names
# %Y%m%d_%H%M%S gives format like 20240115_143052
timestamp=$(date +%Y%m%d_%H%M%S)

Run SD 1.5 on GPU 0
CUDA_VISIBLE_DEVICES=0 python csv_runner.py \
    --model sd15 \
    --t-lo 0.32 \
    --original-csv "./Task/sample_enhanced_prompts.csv" \
    --enhanced-csv "./Task/sample_enhanced_prompts.csv" \
    --prompt-col "prompt" \
    --enhanced-col "modified_prompts" \
    --n-samples ${N} \
    --p-opt-lr 0.0013 \
    --output-dir "./outputs/csv/main" \
    > logs/csv/main/sd15_${timestamp}.log 2>&1 &

pid_sd15=$!
echo "Started SD 1.5 on GPU 0 (PID: ${pid_sd15})"

# Run SD 2.0 on GPU 1
# CUDA_VISIBLE_DEVICES=1 python csv_runner.py \
#     --model sd20 \
#     --t-lo ${t_lo} \
#     --original-csv "./Task/sample_enhanced_prompts.csv" \
#     --enhanced-csv "./Task/sample_enhanced_prompts.csv" \
#     --prompt-col "prompt" \
#     --enhanced-col "modified_prompts" \
#     --n-samples ${N} \
#     --p-opt-lr ${lr} \
#     --output-dir "./outputs/csv/main" \
#     > logs/csv/main/sd20_${timestamp}.log 2>&1 &

# # Store the process ID of the SD 2.0 job
# pid_sd20=$!
# echo "Started SD 2.0 on GPU 1 (PID: ${pid_sd20})"

# --use-lightning flag sets model="sdxl_lightning", method="ddim_lightning", NFE=4, cfg_guidance=1.0
CUDA_VISIBLE_DEVICES=1 python csv_runner.py \
    --use-lightning \
    --t-lo ${t_lo} \
    --original-csv "./Task/original_prompts.csv" \
    --enhanced-csv "./Task/original_prompts.csv" \
    --prompt-col "prompt" \
    --enhanced-col "modified_prompts" \
    --n-samples ${N} \
    --p-opt-lr ${lr} \
    --output-dir "./outputs/csv/main" \
    > logs/csv/main/sdxl_lightning_${timestamp}.log 2>&1 &

# Store the process ID of the SDXL Lightning job
pid_sdxl=$!
echo "Started SDXL Lightning on GPU 1 (PID: ${pid_sdxl})"

# echo ""
# echo "Logs are being written to:"
# echo "  - logs/csv/main/sd15_${timestamp}.log"
# echo "  - logs/csv/main/sd20_${timestamp}.log"
# echo "  - logs/csv/main/sdxl_lightning_${timestamp}.log"
# echo ""
# echo "To monitor logs in real-time:"
# echo "  tail -f logs/csv/main/sd15_${timestamp}.log"
# echo "  tail -f logs/csv/main/sd20_${timestamp}.log"
# echo "  tail -f logs/csv/main/sdxl_lightning_${timestamp}.log"
# echo ""

# Wait for all processes to complete
# wait without arguments waits for all background jobs
# wait ${pid_sd15} ${pid_sdxl}
wait ${pid_sdxl}

echo "All experiments completed!"