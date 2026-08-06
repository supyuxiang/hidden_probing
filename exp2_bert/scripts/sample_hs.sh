#!/usr/bin/env bash
# Sample BERT hiddens for multilingual MATH questions (CLS pooling).
# en uses train.json / test.json; other langs use train_{lang}.json / test_{lang}.json

set -euo pipefail

PYTHON="${PYTHON:-/root/autodl-tmp/miniconda3/envs/bert1/bin/python}"
SCRIPT="/root/hidden_prob/exp2_bert/sample_hs.py"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/bert-base-multilingual-cased}"
DATA_DIR="${DATA_DIR:-/root/hidden_prob/data/math}"
HS_DIR="${HS_DIR:-/root/autodl-tmp/exp2_bert/hs/bert-base-multilingual-cased}"
BATCH_SIZE="${BATCH_SIZE:-64}"
POOLING_MODE="${POOLING_MODE:-cls}"
LAYER_INDICES="${LAYER_INDICES:-all}"
TEXT_FIELD="${TEXT_FIELD:-question}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${HS_DIR}"

run_one() {
    local split="$1"   # train | test
    local lang="$2"    # en | zh | es | vi | tr
    local data_path
    if [ "${lang}" = "en" ]; then
        data_path="${DATA_DIR}/${split}.json"
    else
        data_path="${DATA_DIR}/${split}_${lang}.json"
    fi
    local save_path="${HS_DIR}/hs_math_${split}_${lang}.pt"

    echo "===== ${split}/${lang} ====="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" "${SCRIPT}" \
        --batch_size "${BATCH_SIZE}" \
        --model_path "${MODEL_PATH}" \
        --data_path "${data_path}" \
        --save_path "${save_path}" \
        --layer_indices "${LAYER_INDICES}" \
        --pooling_mode "${POOLING_MODE}" \
        --text_field "${TEXT_FIELD}"
}

######## train ########
run_one train en
run_one train zh
run_one train es
run_one train vi
run_one train tr

######## test (uncomment if needed) ########
# run_one test en
# run_one test zh
# run_one test es
# run_one test vi
# run_one test tr
