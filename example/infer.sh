method="GCG"
PREFIX=./configs/optimizer_
SUFFIX=_config.yaml
TEST_CASE_PREFIX=./output/GCG
config="${PREFIX}${method}${SUFFIX}"

echo "attacking"
CUDA_VISIBLE_DEVICES="0" python example/example.py  \
--config_path $config \
--test_path ./data/test_pubmed.json \
--test_case_path "${TEST_CASE_PREFIX}/${method}_PubMedQA_suffixs.json" \
--dataset_name PubMedQA \
--evaluate

echo "checking result"
python example/example.py  \
--config_path $config \
--test_case_path "${TEST_CASE_PREFIX}/${method}_PubMedQA_suffixs.json" \
--dataset_name PubMedQA \
--check_refuse \
--completion_path "${TEST_CASE_PREFIX}/${method}_PubMedQA_completion.json"