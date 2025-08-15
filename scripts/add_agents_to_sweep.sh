#!/bin/bash

# Default values
MODEL_NAME="baseline"
COUNT=2
AGENTS=1
SWEEP_ID=""

# Function to display usage
usage() {
    echo "Usage: $0 --sweep-id SWEEP_ID [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --sweep-id ID     W&B sweep ID to add agents to"
    echo ""
    echo "Options:"
    echo "  --model NAME    W&B project name (default: baseline)"
    echo "  --count N         Experiments per agent (default: 2)"
    echo "  --agents N        Number of agents to add (default: 1)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --sweep-id abc123def --agents 3 --count 5"
    echo "  $0 --sweep-id xyz789ghi --project my_model --agents 2"
    echo ""
    echo "💡 To find your sweep ID:"
    echo "   - Check your W&B dashboard at https://wandb.ai/AI_RCP/[PROJECT]/sweeps/"
    echo "   - Or look at the output from when you started the sweep"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --count)
            COUNT="$2"
            shift 2
            ;;
        --agents)
            AGENTS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if sweep ID is provided
if [ -z "$SWEEP_ID" ]; then
    echo "❌ Error: Sweep ID is required!"
    echo ""
    usage
    exit 1
fi

echo "🎯 Adding agents to existing sweep"
echo "📊 Sweep ID: $SWEEP_ID"
echo "🏷️  Model: $MODEL_NAME"
echo "🔢 Experiments per agent: $COUNT"
echo "👥 Number of agents to add: $AGENTS"
echo ""

# Create outputs directory
mkdir -p outputs_sweep

# Load environment (needed for W&B authentication check)
ml Miniforge3
conda activate AI_RCP_env_3

# Check if W&B is authenticated (try multiple methods)
echo "🔍 Checking W&B authentication..."

# Method 1: Check for API key in environment or config
WANDB_AUTH_OK=false

# Try to run a simple wandb command to check authentication
if python -c "import wandb; wandb.login()" >/dev/null 2>&1; then
    WANDB_AUTH_OK=true
elif wandb status 2>/dev/null | grep -q "Logged in"; then
    WANDB_AUTH_OK=true
elif [ -f ~/.netrc ] && grep -q "api.wandb.ai" ~/.netrc; then
    WANDB_AUTH_OK=true
elif [ ! -z "$WANDB_API_KEY" ]; then
    WANDB_AUTH_OK=true
fi

if [ "$WANDB_AUTH_OK" = false ]; then
    echo "❌ W&B authentication required. Please run:"
    echo "   bash scripts/setup_wandb_hpc.sh"
    echo "   OR manually: ml Miniforge3 && conda activate AI_RCP_env_3 && wandb login"
    exit 1
fi

echo "✅ W&B authentication verified"
echo ""

# Verify sweep exists by trying to get sweep info
echo "🔍 Verifying sweep exists..."
SWEEP_CHECK=$(python -c "
import wandb
try:
    api = wandb.Api()
    sweep = api.sweep('AI_RCP/$MODEL_NAME/$SWEEP_ID')
    print(f'✅ Found sweep: {sweep.name}')
    print(f'📈 State: {sweep.state}')
    print(f'🔗 URL: https://wandb.ai/AI_RCP/$MODEL_NAME/sweeps/$SWEEP_ID')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
" 2>&1)

if [[ $SWEEP_CHECK == *"❌"* ]]; then
    echo "$SWEEP_CHECK"
    echo ""
    echo "💡 Tips:"
    echo "   - Make sure the sweep ID is correct"
    echo "   - Check that the model name matches: $MODEL_NAME"
    echo "   - Verify the sweep exists at: https://wandb.ai/AI_RCP/$MODEL_NAME/sweeps/"
    exit 1
fi

echo "$SWEEP_CHECK"
echo ""

# Submit the additional agents
echo "🚀 Adding $AGENTS new agents to sweep..."
for i in $(seq 1 $AGENTS); do
    # Generate unique job name with timestamp
    TIMESTAMP=$(date +%H%M%S)
    
    # Create temporary script for this agent
    TEMP_SCRIPT="outputs_sweep/add_agent_${i}_${TIMESTAMP}.sh"
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=add_agent_${i}_${TIMESTAMP}_${MODEL_NAME}
#SBATCH --qos 3d
#SBATCH --partition=batch_gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=outputs_sweep/add_agent_${i}_${TIMESTAMP}_%j.out
#SBATCH --error=outputs_sweep/add_agent_${i}_${TIMESTAMP}_%j.err
#SBATCH --time=16:00:00

echo "##########################################"
echo "Job Started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "##########################################"
echo ""

ml Miniforge3
conda activate AI_RCP_env_3
python run_sweep.py --agent_only $SWEEP_ID --project $MODEL_NAME --entity AI_RCP --count $COUNT

echo ""
echo "##########################################"
echo "Job Finished at: \$(date)"
echo "##########################################"
EOF

    chmod +x "$TEMP_SCRIPT"
    JOB_OUTPUT=$(sbatch "$TEMP_SCRIPT" 2>&1)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+')
    
    if [ -n "$JOB_ID" ]; then
        echo "  ✅ Agent $i submitted (Job ID: $JOB_ID)"
    else
        echo "  ❌ Failed to submit job for agent $i. sbatch output:"
        echo "     $JOB_OUTPUT"
    fi
    
    # Clean up temporary script
    rm "$TEMP_SCRIPT"
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo ""
echo "🎉 Successfully added $AGENTS agents to sweep: $SWEEP_ID"
echo "📊 Monitor sweep: https://wandb.ai/AI_RCP/$MODEL_NAME/sweeps/$SWEEP_ID"
echo "📋 View logs: ls outputs_sweep/"
echo "👀 Check jobs: squeue -u \$USER --name='*add_agent*${MODEL_NAME}*'"
echo ""
echo "💡 You can add more agents anytime by running this script again!" 