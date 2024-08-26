pip install -e .
huggingface-cli login --token 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
make download-everything
git clone https://huggingface.co/impossibleexchange/try2takemyhat
make finetune-x1
export HF_ACCESS_TOKEN=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
python miner_utils/up.py --hf_repo_id cmncomp/a2a_vp --model_dir output_checkpoints/experiment_1/ --epoch 0 
