#!/bin/bash
#SBATCH --job-name=clip_max77_keep
#SBATCH --output=logs/clip_max77_keep_%j.out
#SBATCH --error=logs/clip_max77_keep_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amascare@sissa.it

# Build the length-77 keep mask for the CLIP-text rebuttal A/B.
# CPU-only, tokenises dataset['txt'][:5000] with the CLIP-L tokeniser.
# Output: data/clip_max77_keep.npy (length-5000 bool).

source /home/$USER/myjax/bin/activate
cd /home/$USER/deepseek-hidden-states
mkdir -p logs data

echo "========================================="
echo "Build CLIP max77 keep mask"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "========================================="

python build_clip_max77_keep.py

STATUS=$?
echo ""
echo "Finished at: $(date)"
echo "Exit code: ${STATUS}"
exit ${STATUS}
