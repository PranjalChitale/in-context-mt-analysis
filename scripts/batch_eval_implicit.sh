#/bin/bash

model_name=$1
test_set=$2
aux_lang=$3
src_lang=$4
tgt_lang=$5


dev_fname="data/dev/flores_${aux_lang}-${tgt_lang}.jsonl"
test_fname="data/test/${test_set}_${src_lang}-${tgt_lang}.jsonl"


declare -A batch_sizes
batch_sizes=( [1]=32 [4]=24 [8]=16 )
n_shots=(1 4 8)

# --------------------------------------------------------------------------
#                Implicit (allied tasks proxy) in-context eval
# --------------------------------------------------------------------------
for n_shot in "${n_shots[@]}"; do
    batch_size=${batch_sizes[$n_shot]}
    
    if [ -z "$batch_size" ]; then
        echo "Unsupported n_shot value: $n_shot"
        exit 1
    fi

    echo "Running standard in-context eval for ${src_lang}-${tgt_lang} for $n_shot shot"

    python3 src/causal_lm_eval_implicit.py \
        --model_name_or_path $model_name \
        --tokenizer_name_or_path $model_name \
        --dev_fname $dev_fname --test_fname $test_fname \
        --src_lang $src_lang --tgt_lang $tgt_lang --aux_lang $aux_lang \
        --n_shot $n_shot --seed 42 --batch_size $batch_size

done
