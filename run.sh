pip install -e .
huggingface-cli login --token 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
make download-everything
git clone https://huggingface.co/impossibleexchange/try2takemyhat
make finetune-x1
python miner_utils/up.py --hf_repo_id minh132/test --model_dir output_checkpoints/experiment_1/ --epoch 0

