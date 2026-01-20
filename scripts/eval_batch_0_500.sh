#!/bin/bash
# Batch evaluation script for motion IDs 0-500
# Usage: ./scripts/eval_batch_0_500.sh <config> <checkpoint> <input_pkl> <output_dir_base> [eval_stride]
#
# Example:
#   ./scripts/eval_batch_0_500.sh \
#     configs/agent_codebook_switching.yaml \
#     /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
#     /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
#     ./evaluation_plots_rec_pred_dyn_batch

if [ $# -lt 4 ]; then
    echo "Usage: $0 <config> <checkpoint> <input_pkl> <output_dir_base> [eval_stride]"
    echo "Example: $0 configs/agent_codebook_switching.yaml checkpoints/model.ckpt data.pkl ./evaluation_batch"
    exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
INPUT_PKL="$3"
OUTPUT_DIR_BASE="$4"
EVAL_STRIDE="${5:-}"  # Optional eval_stride

# Validate required files exist
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$INPUT_PKL" ]; then
    echo "Error: Input PKL file not found: $INPUT_PKL"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR_BASE"
LOG_DIR="$OUTPUT_DIR_BASE/logs"
mkdir -p "$LOG_DIR"

# Specific motion IDs to evaluate
FALCON_BETTER_IDS=(351 211 214 305 98 341 285 156 10 303 72 230 213 302 315 207)
GMT_BETTER_IDS=(8 191 140 216 130 133 56 195 65 97 135 192 180 0 66 23)

# Combine both arrays
MOTION_IDS=("${FALCON_BETTER_IDS[@]}" "${GMT_BETTER_IDS[@]}")

echo "=========================================="
echo "Batch Evaluation: Specific Motion IDs"
echo "  FALCON_BETTER_IDS: ${FALCON_BETTER_IDS[*]}"
echo "  GMT_BETTER_IDS: ${GMT_BETTER_IDS[*]}"
echo "  Total: ${#MOTION_IDS[@]} motion IDs"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Input PKL: $INPUT_PKL"
echo "Output Base: $OUTPUT_DIR_BASE"
echo "=========================================="

# Function to run a single evaluation
run_evaluation() {
    local MOTION_ID=$1
    local CONFIG_VAR="$2"
    local CHECKPOINT_VAR="$3"
    local INPUT_PKL_VAR="$4"
    local OUTPUT_DIR_BASE_VAR="$5"
    local EVAL_STRIDE_VAR="$6"
    local LOG_DIR_VAR="$7"
    
    local OUTPUT_DIR="$OUTPUT_DIR_BASE_VAR/motion_${MOTION_ID}"
    local LOG_FILE="$LOG_DIR_VAR/motion_${MOTION_ID}.log"
    
    # Build command as array to avoid quoting issues
    local CMD_ARGS=(
        "python" "scripts/eval_vqvae_rec_pred_dyn.py"
        "--config" "$CONFIG_VAR"
        "--checkpoint" "$CHECKPOINT_VAR"
        "--input_pkl" "$INPUT_PKL_VAR"
        "--motion_id" "$MOTION_ID"
        "--output_dir" "$OUTPUT_DIR"
    )
    
    if [ -n "$EVAL_STRIDE_VAR" ]; then
        CMD_ARGS+=("--eval_stride" "$EVAL_STRIDE_VAR")
    fi
    
    # Run evaluation and capture output
    # Debug: print command for first motion
    if [ "$MOTION_ID" -eq 0 ]; then
        echo "Debug: Running command for motion 0:" >&2
        echo "  ${CMD_ARGS[*]}" >&2
    fi
    
    if "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1; then
        echo "[$MOTION_ID] ✓ Success" >&2
        return 0
    else
        echo "[$MOTION_ID] ✗ Failed (check $LOG_FILE)" >&2
        # Show first few lines of error for motion 0
        if [ "$MOTION_ID" -eq 0 ]; then
            echo "First few lines of error log:" >&2
            head -10 "$LOG_FILE" >&2
        fi
        return 1
    fi
}

# Sequential execution
TOTAL=${#MOTION_IDS[@]}
CURRENT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for MOTION_ID in "${MOTION_IDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Processing motion ID: $MOTION_ID"
    
    if run_evaluation $MOTION_ID "$CONFIG" "$CHECKPOINT" "$INPUT_PKL" "$OUTPUT_DIR_BASE" "$EVAL_STRIDE" "$LOG_DIR"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "=========================================="
echo "Batch Evaluation Complete"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "  Total: $((SUCCESS_COUNT + FAIL_COUNT))"
echo "=========================================="

# Aggregate metrics
echo ""
echo "Aggregating metrics..."
# Calculate min and max for aggregate script (it checks for file existence, so missing IDs are fine)
MIN_ID=$(printf '%s\n' "${MOTION_IDS[@]}" | sort -n | head -1)
MAX_ID=$(printf '%s\n' "${MOTION_IDS[@]}" | sort -n | tail -1)
python scripts/aggregate_batch_metrics.py "$OUTPUT_DIR_BASE" "$MIN_ID" "$MAX_ID"

echo ""
echo "Done! Check $OUTPUT_DIR_BASE/aggregated_metrics.json for average performance."

