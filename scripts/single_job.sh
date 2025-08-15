#!/bin/bash
#SBATCH --job-name=train_RCP
#SBATCH --qos 3d
#SBATCH --partition=batch_gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=outputs/output-%j.out
#SBATCH --error=outputs/error-%j.err
#SBATCH --time=16:00:00

echo "##########################################"
echo "Job Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "##########################################"
echo ""

ml Miniforge3
conda activate AI_RCP_env_3_backup_4

# Your command
#python train_gen_model.py --config_file configs/base_config.yaml
# python train_xgboost_yield_indexed.py --config_file configs/xgboost_config.yaml
python experiment_2.py \
    --config_file best_configs_sm/sm_all_seq_emb.yaml \
    --test_smiles_list "FC(F)[C@@H]1COC(=O)N1c2cn3CCOc4cc(Br)ccc4c3n2.COc1cc(nn2cc(nc12)C)[B]1OC(C)(C)C(C)(C)O1>>COc1cc(nn2cc(nc12)C)-c1ccc-2c(c1)OCCn1cc(nc12)N1[C@@H](COC1=O)C(F)F" \
    --test_injection_percentage 0.2

echo ""
echo "##########################################"
echo "Job Finished at: $(date)"
echo "##########################################"