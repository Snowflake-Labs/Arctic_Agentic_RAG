#!/bin/bash

timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
model=$1  # e.g., gpt4o_azure
dataset=$2  # e.g., msmarco, asqa, or squad
ROOT_DIR=$3  # path to your repo root
limit=${4:-10}  # optional: number of examples to process, default 10

outdir=results/ambiguity_${model}_${dataset}_${timestamp}

# Map dataset names to file paths
case $dataset in
    msmarco)
        input_fname="ambiguous/msmarco-ambiguous.json"
        ;;
    asqa)
        input_fname="ambiguous/asqa.json"
        ;;
    squad)
        input_fname="general/squad-train-v1.1.json"
        ;;
    *)
        echo "Error: Unrecognized dataset '$dataset'. Use msmarco, asqa, or squad"
        exit 1
        ;;
esac

# Create output directory
mkdir -p $outdir

# Add repository to Python path
export PYTHONPATH="$PYTHONPATH:${ROOT_DIR}"

# Run the ambiguity detection
python main.py \
    --agent_config configs/config_ambiguity_${model}.yaml \
    --input_file ./data/${input_fname} \
    --result_path $outdir \
    --limit $limit
