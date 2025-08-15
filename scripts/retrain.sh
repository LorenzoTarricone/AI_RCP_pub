#!/bin/bash

# This script submits retraining jobs to Slurm for each configuration file in a specified folder.
#
# Usage:
# ./scripts/retrain.sh <config_folder>
#
# Example:
# ./scripts/retrain.sh best_configs
#
# The script iterates through all .yaml files in the provided folder,
# and for each config, it generates and submits a Slurm job.

# --- Configuration ---
# Number of times to retrain the model for each configuration.
# You can change this value if needed.
N_TRAININGS=10
LOG_DIR="outputs_retrain"

# --- Script Logic ---
# Check if a configuration folder is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <config_folder>"
  exit 1
fi

CONFIG_FOLDER=$1

# Check if the configuration folder exists
if [ ! -d "$CONFIG_FOLDER" ]; then
  echo "Error: Directory '$CONFIG_FOLDER' not found."
  exit 1
fi

# Create a directory for Slurm output files
mkdir -p "$LOG_DIR"

# Find all .yaml files in the specified folder and iterate through them
find "$CONFIG_FOLDER" -type f -name "*.yaml" | while read -r CONFIG_FILE; do
  if [ -f "$CONFIG_FILE" ]; then
    # Extract a base name from the config file for the job name
    BASENAME=$(basename "$CONFIG_FILE" .yaml)
    
    echo "----------------------------------------------------"
    echo "Submitting training job for config: $CONFIG_FILE"
    echo "Job Name: retrain_${BASENAME}"
    echo "----------------------------------------------------"

    # Use sbatch with a here-document to avoid temporary files
    # This is a cleaner approach than creating and deleting a script file.
    JOB_OUTPUT=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=retrain_${BASENAME}
#SBATCH --qos=3d
#SBATCH --partition=batch_gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=${LOG_DIR}/retrain_${BASENAME}_%j.out
#SBATCH --error=${LOG_DIR}/retrain_${BASENAME}_%j.err
#SBATCH --time=16:00:00

echo "##########################################"
echo "Job Started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Config File: ${CONFIG_FILE}"
echo "##########################################"
echo ""

# Load required modules and activate environment
ml Miniforge3
conda activate AI_RCP_env_3

# Execute the retraining script
python retrain_best_models.py --config_file "${CONFIG_FILE}" --n_trainings "${N_TRAININGS}"

echo ""
echo "##########################################"
echo "Job Finished at: \$(date)"
echo "##########################################"
EOF
)
    
    # Correctly extract the job ID from sbatch output
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        echo "  ✅ Job submitted with ID: $JOB_ID"
    else
        echo "  ❌ Failed to submit job for config: $CONFIG_FILE"
        echo "Sbatch output: $JOB_OUTPUT"
    fi
    
    # Optional: pause between submissions
    sleep 1
  fi
done

echo ""
echo "All training jobs have been submitted to Slurm."
echo "👀 Check job status with: squeue -u \$USER"
echo "📋 View logs in the '${LOG_DIR}' directory."
