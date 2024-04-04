#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=temp
#SBATCH --output=logs/temp-%j.log
#SBATCH --mem=100MB
#SBATCH --partition=regular

# Rename models GNN depth 1 original
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_y_diffpool_1.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_y_diffpool_1_original.rds
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_predictions_diffpool_1.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_predictions_diffpool_1_original.rds
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_diffs_diffpool_1.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_final_diffs_diffpool_1_original.rds
mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_diffpool_1.rds gnn_full/DDD_FREE_TES/DDD_FREE_TES_diffpool_1_original.rds
mv mv gnn_full/DDD_FREE_TES/DDD_FREE_TES_model_diffpool_1.pt gnn_full/DDD_FREE_TES/DDD_FREE_TES_model_diffpool_1_original.pt