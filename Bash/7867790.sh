#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=temp
#SBATCH --output=logs/temp-%j.log
#SBATCH --mem=100MB
#SBATCH --partition=regular

# Rename models GNN depth 2 stacking
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_y_diffpool_2.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_y_diffpool_2_stacking.rds
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_predictions_diffpool_2.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_predictions_diffpool_2_stacking.rds
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_diffs_diffpool_2.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_diffs_diffpool_2_stacking.rds
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_diffpool_2.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_diffpool_2_stacking.rds
mv mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_model_diffpool_2.pt gnn_full/DDD_FREE_TES/DDD_FREE_TES_model_diffpool_2_stacking.pt