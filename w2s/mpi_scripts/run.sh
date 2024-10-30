source ~/miniforge3/etc/profile.d/conda.sh # equivalent to conda init
conda activate diff
export HOME=/home/sgoel
# optionally parse args
arg_1="$1"
arg_2="$2"

# execute python script
python run.py --dataset="$arg_1" --run_name="$arg_2"