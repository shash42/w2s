source ~/miniforge3/etc/profile.d/conda.sh # equivalent to conda init
conda activate diff
export HOME=/home/sgoel
export SOFT_FILELOCK=1
# optionally parse args
dataset="$1"
run_name="$2"
weak_model_name="$3"
strong_model_name="$4"

# execute python script
python run.py --dataset="$dataset" --run_name="$run_name" --weak_model_name="$weak_model_name" --strong_model_name="$strong_model_name"