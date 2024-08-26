pip install -e .
huggingface-cli login --token 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
make download-everything
git clone https://huggingface.co/impossibleexchange/try2takemyhat
make finetune-x1
