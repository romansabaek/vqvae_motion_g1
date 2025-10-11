#!/bin/bash

# Motion VQVAE Evaluation Script
# This script runs motion generation and evaluation for different sets of motion IDs

# Set default values
CONFIG_FILE="configs/agent.yaml"
CHECKPOINT_FILE="outputs/run_0_300/best_model.ckpt"
INPUT_PKL="/home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl"
OUTPUT_DIR="./outputs"
DEVICE="auto"

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_FILE"
    exit 1
fi

if [ ! -f "$INPUT_PKL" ]; then
    echo "Error: Input PKL file not found: $INPUT_PKL"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🎬 Motion VQVAE Evaluation"
echo "📁 Output: $OUTPUT_DIR"
echo "📋 Config: $CONFIG_FILE"
echo "🔧 Checkpoint: $CHECKPOINT_FILE"
echo "📊 Input: $(basename $INPUT_PKL)"

# Function to run evaluation for a range of motion IDs
run_evaluation() {
    local start_id="$1"
    local end_id="$2"
    local description="$3"
    local csv_file="$OUTPUT_DIR/${description}_evaluation.csv"
    
    echo "🚀 $description (motions: $start_id-$end_id) ..."
    echo "📊 CSV: $csv_file"
    
    # Calculate total motions
    total_motions=$((end_id - start_id + 1))
    current_motion=0
    
    # Run Python script for each motion ID
    for motion_id in $(seq $start_id $end_id); do
        current_motion=$((current_motion + 1))
        
        # Show progress
        progress=$((current_motion * 100 / total_motions))
        printf "\r  📊 Progress: [%3d%%] (%d/%d) Processing motion %d..." $progress $current_motion $total_motions $motion_id
        
        # Run with minimal output
        python scripts/generate_motion_from_vqvae.py \
            --config "$CONFIG_FILE" \
            --checkpoint "$CHECKPOINT_FILE" \
            --input_pkl "$INPUT_PKL" \
            --motion_ids "$motion_id" \
            --output_file "$OUTPUT_DIR/vqvae_motion_${motion_id}.pkl" \
            --csv_file "$csv_file" \
            >/dev/null 2>&1
    done
    
    # Clear progress line and show completion
    printf "\r  ✅ Progress: [100%%] (%d/%d) Completed!                    \n" $total_motions $total_motions
    
    echo "✅ $description completed"
}

# Run motion evaluations
echo ""
echo "Starting motion evaluations..."

# Generate motions (change the range as needed)
run_evaluation 0 100 "motion_evaluation"

echo ""
echo "🎉 All evaluations completed!"
echo ""
echo "📁 Generated files:"
pkl_count=$(ls "$OUTPUT_DIR"/*.pkl 2>/dev/null | wc -l)
echo "  📂 $pkl_count individual PKL files in $OUTPUT_DIR"
echo ""
echo "📊 CSV evaluation files:"
for csv_file in "$OUTPUT_DIR"/*_evaluation.csv; do
    if [ -f "$csv_file" ]; then
        csv_name=$(basename "$csv_file")
        motion_count=$(tail -n +2 "$csv_file" | wc -l 2>/dev/null || echo "0")
        echo "  📈 $csv_name: $motion_count motions"
    fi
done
echo ""
echo "💡 Each motion has its own PKL file for MuJoCo"
echo "💡 CSV files contain evaluation metrics for each set"
