#!/bin/bash
#
# Random Hyperparameter Search for MinorityPrompt
#
# Runs experiments with random parameters within specified ranges.
#
# Usage:
#   ./run_search.sh                           # Use defaults
#   ./run_search.sh --n-runs 10               # Run 10 experiments
#   ./run_search.sh --lr-min 0.001 --lr-max 0.1
#   ./run_search.sh --iter-min 5 --iter-max 20
#   ./run_search.sh --dry-run                 # Show params without running
#
# Example:
#   ./run_search.sh --n-runs 5 --lr-min 0.001 --lr-max 0.05 --iter-min 5 --iter-max 15

set -e

# =============================================================================
# Default Ranges
# =============================================================================

N_RUNS=1
DRY_RUN=false

# p_opt_iter range (integer)
ITER_MIN=5
ITER_MAX=15

# p_opt_lr range (float)
LR_MIN=0.005
LR_MAX=0.02

# t_lo range (float)
TLO_MIN=0.0
TLO_MAX=0.5

# n_samples range (integer)
SAMPLES_MIN=5
SAMPLES_MAX=10

# seed_start range (integer)
SEED_MIN=0
SEED_MAX=1000

# init_type options (space-separated)
INIT_TYPES="default gaussian"

# model options
MODELS="sdxl"

# Fixed parameters
ORIGINAL_CSV="sample_original_prompts.csv"
ENHANCED_CSV=""
USE_LIGHTNING=false
OUTPUT_DIR="./outputs"

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-runs)       N_RUNS="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        
        # Ranges
        --iter-min)     ITER_MIN="$2"; shift 2 ;;
        --iter-max)     ITER_MAX="$2"; shift 2 ;;
        --lr-min)       LR_MIN="$2"; shift 2 ;;
        --lr-max)       LR_MAX="$2"; shift 2 ;;
        --tlo-min)      TLO_MIN="$2"; shift 2 ;;
        --tlo-max)      TLO_MAX="$2"; shift 2 ;;
        --samples-min)  SAMPLES_MIN="$2"; shift 2 ;;
        --samples-max)  SAMPLES_MAX="$2"; shift 2 ;;
        --seed-min)     SEED_MIN="$2"; shift 2 ;;
        --seed-max)     SEED_MAX="$2"; shift 2 ;;
        
        # Options (space-separated list)
        --init-types)   INIT_TYPES="$2"; shift 2 ;;
        --models)       MODELS="$2"; shift 2 ;;
        
        # Fixed
        --original-csv) ORIGINAL_CSV="$2"; shift 2 ;;
        --enhanced-csv) ENHANCED_CSV="$2"; shift 2 ;;
        --use-lightning) USE_LIGHTNING=true; shift ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run Options:"
            echo "  --n-runs N          Number of experiments to run (default: 1)"
            echo "  --dry-run           Print parameters without running"
            echo ""
            echo "Parameter Ranges:"
            echo "  --iter-min N        Min p_opt_iter (default: 5)"
            echo "  --iter-max N        Max p_opt_iter (default: 15)"
            echo "  --lr-min F          Min p_opt_lr (default: 0.005)"
            echo "  --lr-max F          Max p_opt_lr (default: 0.02)"
            echo "  --tlo-min F         Min t_lo (default: 0.0)"
            echo "  --tlo-max F         Max t_lo (default: 0.5)"
            echo "  --samples-min N     Min n_samples (default: 5)"
            echo "  --samples-max N     Max n_samples (default: 10)"
            echo "  --seed-min N        Min seed_start (default: 0)"
            echo "  --seed-max N        Max seed_start (default: 1000)"
            echo ""
            echo "Parameter Options (space-separated):"
            echo "  --init-types LIST   Init types (default: \"default gaussian\")"
            echo "  --models LIST       Models (default: \"sdxl\")"
            echo ""
            echo "Fixed Parameters:"
            echo "  --original-csv F    Prompts CSV (default: sample_original_prompts.csv)"
            echo "  --enhanced-csv F    Enhanced prompts CSV"
            echo "  --use-lightning     Use SDXL Lightning"
            echo "  --output-dir D      Output directory (default: ./outputs)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Random Functions
# =============================================================================

# Random integer between min and max (inclusive)
rand_int() {
    local min=$1 max=$2
    echo $((min + RANDOM % (max - min + 1)))
}

# Random float between min and max
rand_float() {
    local min=$1 max=$2
    awk -v min="$min" -v max="$max" -v seed="$RANDOM" '
        BEGIN {
            srand(seed)
            printf "%.4f", min + rand() * (max - min)
        }'
}

# Random choice from space-separated list
rand_choice() {
    local options=($1)
    local n=${#options[@]}
    echo "${options[$((RANDOM % n))]}"
}

# =============================================================================
# Main
# =============================================================================

echo "============================================================"
echo "  Random Hyperparameter Search"
echo "============================================================"
echo ""
echo "Ranges:"
echo "  p_opt_iter: [$ITER_MIN, $ITER_MAX]"
echo "  p_opt_lr:   [$LR_MIN, $LR_MAX]"
echo "  t_lo:       [$TLO_MIN, $TLO_MAX]"
echo "  n_samples:  [$SAMPLES_MIN, $SAMPLES_MAX]"
echo "  seed_start: [$SEED_MIN, $SEED_MAX]"
echo "  init_types: $INIT_TYPES"
echo "  models:     $MODELS"
echo ""
echo "Running $N_RUNS experiment(s)..."
echo ""

# Create log file
SEARCH_LOG="$OUTPUT_DIR/search_log_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p "$OUTPUT_DIR"

# Write CSV header
echo "run,model,p_opt_iter,p_opt_lr,t_lo,init_type,n_samples,seed_start,experiment_dir" > "$SEARCH_LOG"

for ((run=1; run<=N_RUNS; run++)); do
    echo "============================================================"
    echo "  Run $run / $N_RUNS"
    echo "============================================================"
    
    # Sample random parameters
    P_OPT_ITER=$(rand_int $ITER_MIN $ITER_MAX)
    P_OPT_LR=$(rand_float $LR_MIN $LR_MAX)
    T_LO=$(rand_float $TLO_MIN $TLO_MAX)
    N_SAMPLES=$(rand_int $SAMPLES_MIN $SAMPLES_MAX)
    SEED_START=$(rand_int $SEED_MIN $SEED_MAX)
    INIT_TYPE=$(rand_choice "$INIT_TYPES")
    MODEL=$(rand_choice "$MODELS")
    
    echo ""
    echo "Parameters:"
    echo "  model:       $MODEL"
    echo "  p_opt_iter:  $P_OPT_ITER"
    echo "  p_opt_lr:    $P_OPT_LR"
    echo "  t_lo:        $T_LO"
    echo "  init_type:   $INIT_TYPE"
    echo "  n_samples:   $N_SAMPLES"
    echo "  seed_start:  $SEED_START"
    echo ""
    
    # Build command
    CMD="python concat_generate_and_evaluate.py"
    CMD="$CMD --original-csv \"$ORIGINAL_CSV\""
    CMD="$CMD --model $MODEL"
    CMD="$CMD --p-opt-iter $P_OPT_ITER"
    CMD="$CMD --p-opt-lr $P_OPT_LR"
    CMD="$CMD --t-lo $T_LO"
    CMD="$CMD --init-type $INIT_TYPE"
    CMD="$CMD --n-samples $N_SAMPLES"
    CMD="$CMD --seed-start $SEED_START"
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
    
    if [[ -n "$ENHANCED_CSV" ]]; then
        CMD="$CMD --enhanced-csv \"$ENHANCED_CSV\""
    fi
    
    if [[ "$USE_LIGHTNING" == true ]]; then
        CMD="$CMD --use-lightning"
    fi
    
    echo "Command:"
    echo "  $CMD"
    echo ""
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Skipping execution"
        EXP_DIR="(dry-run)"
    else
        # Run experiment
        eval $CMD
        
        # Find the most recent experiment directory
        EXP_DIR=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
    fi
    
    # Log to CSV
    echo "$run,$MODEL,$P_OPT_ITER,$P_OPT_LR,$T_LO,$INIT_TYPE,$N_SAMPLES,$SEED_START,$EXP_DIR" >> "$SEARCH_LOG"
    
    echo ""
done

echo "============================================================"
echo "  Search Complete!"
echo "============================================================"
echo ""
echo "Log saved to: $SEARCH_LOG"
echo ""
cat "$SEARCH_LOG"