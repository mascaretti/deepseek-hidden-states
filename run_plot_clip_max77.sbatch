#!/bin/bash
#SBATCH --job-name=plot_clip_max77
#SBATCH --output=logs/plot_clip_max77_%j.out
#SBATCH --error=logs/plot_clip_max77_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it

# Render paper/figs-tmlr/rebuttal/clip_max77.{pdf,png} from the eight II
# safetensors trees produced by the rebuttal A/B.

source /home/$USER/myjax/bin/activate
cd /home/$USER/deepseek-hidden-states

echo "Started at: $(date) on $(hostname)"
python notebooks/clip_max77_plot.py \
    --results_dir /home/laios/image-information-imbalance/results-mean \
    --output paper/figs-tmlr/rebuttal/clip_max77

STATUS=$?
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
exit ${STATUS}
