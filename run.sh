pip install -e .
make download-everything
git clone https://huggingface.co/impossibleexchange/try2takemyhat
make finetune-x1
