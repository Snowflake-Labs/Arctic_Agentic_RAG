timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

path=$1
metric=$2
logfile=$1/eval_run_${timestamp}.log

python evaluation.py \
    --result_path "$path" \
    --result_filename output.json \
    --azure_endpoint $AZURE_ENDPOINT \
    --azure_model $AZURE_GPT4O \
    --metric "$metric" 2>&1 | tee "$logfile"
