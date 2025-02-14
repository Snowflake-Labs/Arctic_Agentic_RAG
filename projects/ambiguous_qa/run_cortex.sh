timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

model=${1:?Error: model is required}
topk=${2:-5}
embedding=${3:-"qa"}
clustering=${4:-"hdbscan"}

outdir="results/cortex_${model}_${clustering}_${embedding}_topk${topk}_${timestamp}"
mkdir -p "$outdir"
logfile="${outdir}/run.log"

python vd.py \
    --num_examples 10 \
    --model $model \
    --backend snowflake \
    --topk $topk \
    --clustering $clustering \
    --encoder_name_or_path cortex_768 \
    --embedding $embedding \
    --output_path "$outdir" 2>&1 | tee "$logfile"
