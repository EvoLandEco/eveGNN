#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=gnn_eve_pars_free
#SBATCH --output=logs/gnn_eve_pars_free-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript -e 'install.packages("devtools", repos="http://cran.us.r-project.org")'
Rscript -e 'devtools::install_github("EvoLandEco/eveGNN@multimodal-stacking-boosting")'
Rscript -e 'devtools::install_github("HHildenbrandt/evesim@tianjian")'
Rscript ../Script/eve_pars_est_free_data.R ${name}