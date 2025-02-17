timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

model_abbv=${1:?Error: model_abbv is required}
topk=${2:-5}
embedding=${3:-"qa"}
clustering=${4:-"hdbscan"}

if [[ "$model_abbv" == "gpt4o" ]]; then
    model=$AZURE_GPT4O
elif [[ "$model_abbv" == "gpt4o-mini" ]]; then
    model=$AZURE_GPT4O_MINI
else
    echo "Error: Unsupported model abbreviation '$model_abbv'"
    exit 1
fi

outdir="results/azure_${model_abbv}_${clustering}_${embedding}_topk${topk}_${timestamp}"
mkdir -p "$outdir"
logfile="${outdir}/run.log"

python vd.py \
    --num_examples 10 \
    --model $model \
    --backend azure \
    --topk $topk \
    --clustering $clustering \
    --encoder_name_or_path cortex_768 \
    --embedding $embedding \
    --azure_endpoint $AZURE_ENDPOINT \
    --output_path "$outdir" 2>&1 | tee "$logfile"
