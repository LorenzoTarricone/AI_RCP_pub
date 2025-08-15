#!/bin/bash

# Default values
COUNT=2           # Experiments per agent
AGENTS=10         # Number of parallel agents
REACTION_TYPE="bh"
DATASET_TYPE="all"
MODEL_TYPE="emb"  # This will also be the default project name

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --model TYPE      Model type to use. This will also be used as the W&B project name."
    echo "                    (options: rxnfp, baseline, seq, emb, seq_emb; default: emb)"
    echo "  --count N         Experiments per agent (default: 2)"
    echo "  --agents N        Number of parallel agents (default: 10)"
    echo "  --reaction TYPE   Reaction type: bh or sm (default: bh)"
    echo "  --dataset TYPE    Dataset type: all or positive (default: all)"
    echo "  -h, --help        Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --count) COUNT="$2"; shift 2 ;;
        --agents) AGENTS="$2"; shift 2 ;;
        --reaction) REACTION_TYPE="$2"; shift 2 ;;
        --dataset) DATASET_TYPE="$2"; shift 2 ;;
        --model) MODEL_TYPE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# --- ENFORCE CONSISTENCY ---
# The project name is always the same as the model type
PROJECT_NAME=$MODEL_TYPE

# Validate model type
if [[ "$MODEL_TYPE" != "rxnfp" && "$MODEL_TYPE" != "baseline" && "$MODEL_TYPE" != "seq" && "$MODEL_TYPE" != "emb" && "$MODEL_TYPE" != "seq_emb" ]]; then
    echo "❌ Error: Invalid model type '$MODEL_TYPE'."
    usage
    exit 1
fi

# Validate reaction type
if [[ "$REACTION_TYPE" != "bh" && "$REACTION_TYPE" != "sm" ]]; then
    echo "❌ Error: Invalid reaction type '$REACTION_TYPE'. Must be 'bh' or 'sm'"
    usage
    exit 1
fi

# Validate dataset type
if [[ "$DATASET_TYPE" != "all" && "$DATASET_TYPE" != "positive" ]]; then
    echo "❌ Error: Invalid dataset type '$DATASET_TYPE'. Must be 'all' or 'positive'"
    usage
    exit 1
fi

# Construct the data file path
DATA_FILE="data/${REACTION_TYPE}_treshold_all_${DATASET_TYPE}_processed.npz"

echo "🎯 Project: $PROJECT_NAME"
echo "🤖 Model type: $MODEL_TYPE"
echo "👥 Number of agents: $AGENTS"
echo "🔢 Experiments per agent: $COUNT"
echo "🧪 Reaction type: $REACTION_TYPE"
echo "📊 Dataset type: $DATASET_TYPE"
echo "📁 Data file: $DATA_FILE"
echo ""

# Create outputs directory
mkdir -p outputs_sweep

# Load environment
ml Miniforge3
conda activate AI_RCP_env_3

# Check if W&B is authenticated
echo "🔍 Checking W&B authentication..."
# --- FIX: Use a more robust, non-interactive authentication check ---
# 'wandb status' returns a non-zero exit code if not logged in.
# We redirect all output to /dev/null to keep the check silent on success.
if ! wandb status &> /dev/null; then
    echo "❌ W&B authentication required. Please log in interactively first by running:"
    echo "   wandb login"
    exit 1
fi
echo "✅ W&B authentication verified"
echo ""

# --- IMMUTABLE PER-SWEEP CONFIG ---
# This approach creates a unique, temporary sweep config file for each run,
# which is the only safe way to handle parallel job submissions.
BASE_SWEEP_CONFIG="configs/sweep_config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UNIQUE_SWEEP_CONFIG="configs/sweep_config_${PROJECT_NAME}_${TIMESTAMP}_${RANDOM}.yaml"

if [ ! -f "$BASE_SWEEP_CONFIG" ]; then
    echo "❌ Error: Base sweep config file not found at $BASE_SWEEP_CONFIG"
    exit 1
fi

# Function to cleanup the unique sweep config on exit
cleanup() {
    if [ -f "$UNIQUE_SWEEP_CONFIG" ]; then
        echo ""
        echo "🧹 Removing temporary sweep config: $UNIQUE_SWEEP_CONFIG"
        rm "$UNIQUE_SWEEP_CONFIG"
    fi
}
trap cleanup EXIT

echo "📝 Creating unique sweep config for this run..."
# Copy the base sweep config to a unique file
cp "$BASE_SWEEP_CONFIG" "$UNIQUE_SWEEP_CONFIG"

# Replace placeholders in the unique sweep config file
sed -i "s/PROJECT_PLACEHOLDER/$PROJECT_NAME/g" "$UNIQUE_SWEEP_CONFIG"
sed -i "s/MODEL_TYPE_PLACEHOLDER/$MODEL_TYPE/g" "$UNIQUE_SWEEP_CONFIG"
sed -i "s|FILEPATH_PLACEHOLDER|$DATA_FILE|g" "$UNIQUE_SWEEP_CONFIG" # Use | as delimiter for file path

echo "✅ Unique sweep config file created: $UNIQUE_SWEEP_CONFIG"
echo ""

# Initialize the sweep using the unique config
# The --project flag here is redundant as it's in the YAML, but it's good for clarity.
SWEEP_OUTPUT=$(python run_sweep.py --init_only --project "$PROJECT_NAME" --sweep_config "$UNIQUE_SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o "Sweep ID: [a-zA-Z0-9]*" | cut -d' ' -f3)

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Failed to initialize sweep"
    # The trap will handle cleanup
    exit 1
fi

echo "✅ Sweep initialized with ID: $SWEEP_ID"
echo "📊 Monitor at: https://wandb.ai/AI_RCP/$PROJECT_NAME/sweeps/$SWEEP_ID"
echo ""

# Submit multiple parallel agents
echo "🏃 Submitting $AGENTS parallel agents..."
for i in $(seq 1 $AGENTS); do
    TEMP_SCRIPT="outputs_sweep/agent_${i}_${PROJECT_NAME}_$(date +%s).sh"
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${PROJECT_NAME}_agent_${i}
#SBATCH --qos=3d
#SBATCH --partition=batch_gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=outputs_sweep/agent_${i}_%j.out
#SBATCH --error=outputs_sweep/agent_${i}_%j.err
#SBATCH --time=16:00:00

echo "##########################################"
echo "Job Started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "##########################################"
echo ""

ml Miniforge3
conda activate AI_RCP_env_3

# The agent gets all its config from the W&B server, including project, model_type etc.
python run_sweep.py --agent_only $SWEEP_ID --project $PROJECT_NAME --entity AI_RCP --count $COUNT

echo ""
echo "##########################################"
echo "Job Finished at: \$(date)"
echo "##########################################"
EOF

    chmod +x "$TEMP_SCRIPT"
    JOB_OUTPUT=$(sbatch "$TEMP_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o "[0-9]*")
    echo "  ✅ Agent $i submitted (Job ID: $JOB_ID)"
    rm "$TEMP_SCRIPT"
    sleep 1
done

echo ""
echo "🎉 All $AGENTS agents submitted for sweep ID: $SWEEP_ID"
echo "👀 Check jobs: squeue -u \$USER --name='*${PROJECT_NAME}*'"
echo "📋 View logs: ls outputs_sweep/"

# The 'trap' will automatically call the cleanup function on exit.
