source ~/miniforge3/etc/profile.d/conda.sh # equivalent to conda init
conda activate diff

# optionally parse args
arg_1=$1
arg_2=$2

# execute python script
cd ..
python run.py --dataset=$arg_1 --run_name=$arg_2