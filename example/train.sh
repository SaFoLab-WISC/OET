method="GCG"
PREFIX=./configs/optimizer_
SUFFIX=_config.yaml
TEST_CASE_PREFIX=./output/GCG
config="${PREFIX}${method}${SUFFIX}"

CUDA_VISIBLE_DEVICES="0,2" python example/example.py \
--config_path $config \
--train_path ./data/train_pubmed.json \
--test_case_path "${TEST_CASE_PREFIX}/${method}_PubMedQA_suffixs.json" \
--dataset_name PubMedQA \
--train