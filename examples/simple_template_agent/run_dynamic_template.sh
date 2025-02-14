timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

ROOT_DIR=$1 # your local path to the root directory
model="llama3.3-70b"
outdir="results/dynamic_template_${model}_${timestamp}"

mkdir -p $outdir

export PYTHONPATH="$PYTHONPATH:${ROOT_DIR}"
python main_dynamic.py \
  --agent_config "configs/config_dynamic-template_${model}.yaml" \
  --input_file "data/crag-subset.json" \
  --template_config "configs/dynamic_template_example.yaml" \
  --result_path "$outdir" \
  --root_dir "$ROOT_DIR"
