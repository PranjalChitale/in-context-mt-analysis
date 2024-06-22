#/bin/bash

model_name=$1
test_set=$2
src_lang=$3
tgt_lang=$4

if [[ $src_lang == "eng_Latn" ]]; then
    dev_fname="data/dev/flores_${src_lang}-${tgt_lang}.jsonl"
    test_fname="data/test/${test_set}_${src_lang}-${tgt_lang}.jsonl"
else
    dev_fname="data/dev/flores_${tgt_lang}-${src_lang}.jsonl"
    test_fname="data/test/${test_set}_${tgt_lang}-${src_lang}.jsonl"
fi

declare -A batch_sizes
batch_sizes=( [1]=16 [4]=8 [8]=8 )
n_shots=(4 8)
attack_types=("span_noise" "word_duplication" "ocr" "word_order")
noise_percentages=(0.1 0.25 0.5 0.75)
types=("I" "II" "III" "IV")

# --------------------------------------------------------------------------
#                     Heterogeneous Perturbed in-context eval
# --------------------------------------------------------------------------

for n_shot in "${n_shots[@]}"; do
    batch_size=${batch_sizes[$n_shot]}
    
    if [ -z "$batch_size" ]; then
        echo "Unsupported n_shot value: $n_shot"
        exit 1
    fi

    for attack_type in "${attack_types[@]}"; do
        for noise_percentage in "${noise_percentages[@]}"; do

            for type in "${types[@]}"; do

                echo "Running perturbed in-context eval for ${src_lang}-${tgt_lang} for ${n_shot} shot"
                
                # source side perturbation
                python3 src/causal_lm_eval_perturb_hetero.py \
                    --model_name_or_path $model_name \
                    --tokenizer_name_or_path $model_name \
                    --dev_fname $dev_fname \
                    --test_fname $test_fname \
                    --src_lang $src_lang --tgt_lang $tgt_lang \
                    --n_shot $n_shot --batch_size $batch_size --max_new_tokens 256 \
                    --perturb --perturb_direction source \
                    --heterogeneous \
                    --heterogeneous_type $type \
                    --attack_type $attack_type --noise_percentage $noise_percentage
                
                # target side perturbation
                python3 src/causal_lm_eval_perturb_hetero.py \
                    --model_name_or_path $model_name \
                    --tokenizer_name_or_path $model_name \
                    --dev_fname $dev_fname \
                    --test_fname $test_fname \
                    --src_lang $src_lang --tgt_lang $tgt_lang \
                    --n_shot $n_shot --batch_size $batch_size --max_new_tokens 256 \
                    --perturb --perturb_direction target \
                    --heterogeneous \
                    --heterogeneous_type $type \
                    --attack_type $attack_type --noise_percentage $noise_percentage
            done
        done
    done
done
