#!/bin/bash

echo "🔧 Setting up W&B for HPC sweep..."

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "❌ wandb is not installed. Please install it first:"
    echo "   pip install wandb"
    exit 1
fi

# Check if already logged in
if wandb status | grep -q "Logged in"; then
    echo "✅ Already logged into W&B!"
    wandb status
else
    echo "🔐 Logging into W&B..."
    wandb login
    if [ $? -eq 0 ]; then
        echo "✅ W&B login successful!"
    else
        echo "❌ W&B login failed. Please check your API key."
        exit 1
    fi
fi

# Create necessary directories
mkdir -p outputs_sweep
mkdir -p scripts

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "✅ Setup complete! W&B is ready for HPC."
echo "ℹ️  Note: Connection tests may fail on HPC login nodes but will work on compute nodes."
echo ""
echo "📋 Next steps:"
echo "   Option 1 (Recommended): scripts/submit_sweep_wrapper.sh --model emb"
echo "   Option 2 (Simple):      scripts/single_sweep_wrapper.sh --model emb"
echo "   Option 3 (Advanced):    sbatch scripts/submit_sweep.sh  (hardcoded project)"
echo "   Option 4 (Add agents):  scripts/add_agents_to_sweep.sh --sweep-id YOUR_SWEEP_ID"
echo ""
echo "💡 Examples:"
echo "   # Basic usage with default settings (emb model, bh reaction, all data)"
echo "   scripts/submit_sweep_wrapper.sh --model emb --count 3 --agents 5"
echo "   scripts/single_sweep_wrapper.sh --model emb --count 15"
echo ""
echo "   # Specify model type, reaction type, and dataset type (project name = model type)"
echo "   scripts/submit_sweep_wrapper.sh --model rxnfp --reaction bh --dataset all --agents 4"
echo "   scripts/submit_sweep_wrapper.sh --model seq_emb --reaction sm --dataset positive --agents 8"
echo "   scripts/single_sweep_wrapper.sh --model baseline --reaction sm --dataset all --count 20"
echo ""
echo "   # Available model types: rxnfp, baseline, seq, emb, seq_emb"
echo "   # Available reaction types: bh (Buchwald-Hartwig), sm (Suzuki-Miyaura)"
echo "   # Available dataset types: all (positive+negative), positive (positive only)"
echo ""
echo "   # Add more agents to existing sweep"
echo "   scripts/add_agents_to_sweep.sh --sweep-id abc123def --agents 3"
echo ""
echo "🎯 Your sweep will appear at: https://wandb.ai/AI_RCP/[MODEL_TYPE]/sweeps/"
echo ""
echo "🔍 Monitor jobs with:"
echo "   squeue -u \$USER"
echo "   tail -f outputs_sweep/sweep-*.out" 