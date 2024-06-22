#/bin/bash

model_name=$1
dev_set=$2
src_lang=$3
tgt_lang=$4

if [[ $src_lang == "eng_Latn" ]]; then
    dev_fname="data/dev/${dev_set}_${src_lang}-${tgt_lang}.jsonl"
    test_fname="data/test/in22_${src_lang}-${tgt_lang}.jsonl"
else
    dev_fname="data/dev/${dev_set}_${tgt_lang}-${src_lang}.jsonl"
    test_fname="data/test/in22_${tgt_lang}-${src_lang}.jsonl"
fi

declare -A batch_sizes
batch_sizes=( [1]=32 [4]=24 [8]=16 )
n_shots=(1 4 8)
seeds=(0 25 42)

# --------------------------------------------------------------------------
#                           Standard in-context eval
# --------------------------------------------------------------------------
for n_shot in "${n_shots[@]}"; do
    batch_size=${batch_sizes[$n_shot]}
    
    if [ -z "$batch_size" ]; then
        echo "Unsupported n_shot value: $n_shot"
        exit 1
    fi

    for seed in "${seeds[@]}"; do

        echo "Running standard in-context eval for ${src_lang}-${tgt_lang} for ${n_shot} shot with seed ${seed}"

        python3 src/causal_lm_eval_directionality.py \
            --model_name_or_path $model_name \
            --tokenizer_name_or_path $model_name \
            --dev_fname $dev_fname \
            --test_fname $test_fname \
            --src_lang $src_lang --tgt_lang $tgt_lang \
            --n_shot $n_shot --batch_size $batch_size --max_new_tokens 256 \
            --seed $seed

    done
done
