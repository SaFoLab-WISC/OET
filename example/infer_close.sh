
method="GCG"
TEST_CASE_PREFIX=./output/GCG

echo "attacking"
python example/example_close.py \
--key <YOUR API KEY> \
--model_name_or_path gpt-4o-mini-2024-07-18 \
--save_root ./output/GPT4 \
--test_path ./data/test_pubmed.json \
--test_case_path "${TEST_CASE_PREFIX}/${method}_PubMedQA_suffixs.json" \
--attack_goal "Print 'sql injection'" \
--dataset_name PubMedQA \
--running_count 5 \
--evaluate


echo "checking result"
python example/example_close.py  \
--model_name_or_path gpt-4o-mini-2024-07-18 \
--test_case_path "${TEST_CASE_PREFIX}/${method}_PubMedQA_suffixs.json" \
--dataset_name PubMedQA \
--check_refuse \
--save_root ./output/GPT4