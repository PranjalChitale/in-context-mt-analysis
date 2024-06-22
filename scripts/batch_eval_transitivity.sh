#/bin/bash

model_name=$1
test_set=$2
src_lang=$3
pivot_lang=$4
tgt_lang=$5


src_to_pivot_fname="data/test/${test_set}_${pivot_lang}-${src_lang}.jsonl"
pivot_to_tgt_fname="data/test/${test_set}_${pivot_lang}-${tgt_lang}.jsonl"

variants=("I" "II" "III" "IV")


# --------------------------------------------------------------------------
#                           Transitivity in-context eval
# --------------------------------------------------------------------------
for variant in "${variants[@]}"; do

    echo "Running standard in-context eval for ${src_lang}-${tgt_lang}"

    python3 causal_lm_eval_transitivity.py \
        --model_name_or_path $model_name \
        --tokenizer_name_or_path $model_name \
        --src_to_pivot_fname $src_to_pivot_fname \
        --pivot_to_tgt_fname $pivot_to_tgt_fname \
        --pivot_lang $pivot_lang --src_lang $src_lang --tgt_lang $tgt_lang \
        --batch_size 16 --max_new_tokens 256 \
        --variant $variant --seed 42

done
