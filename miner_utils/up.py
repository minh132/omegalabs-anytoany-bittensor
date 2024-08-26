import asyncio
import os
import argparse
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore, get_required_files, check_config
from huggingface_hub import update_repo_visibility
import bittensor as bt

def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="The directory of the model to load.",
    )

    parser.add_argument(
        "--epoch",
        type=str,
        help="The epoch number to load e.g. if you want to upload meta_model_0.pt, epoch should be 0",
    )

    # Parse the arguments and create a configuration namespace
    return parser.parse_args()

def validate_repo(ckpt_dir, epoch):
    for filename in get_required_files(epoch):
        filepath = os.path.join(ckpt_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file {filepath} not found in {ckpt_dir}")
    check_config(ckpt_dir)

async def main(config):
    # Get the repo namespace and name
    repo_namespace, repo_name = config.hf_repo_id.split("/")

    # Validate model directory
    validate_repo(config.model_dir, config.epoch)

    # Upload model to Hugging Face
    remote_model_store = HuggingFaceModelStore()
    bt.logging.info(f"Uploading model to Hugging Face: {repo_namespace}/{repo_name}")

    # Upload the model
    await remote_model_store.upload_model(
        model_dir=config.model_dir,
        repo_id=config.hf_repo_id
    )

    # Set repository visibility to public
    try:
        update_repo_visibility(
            repo_id=config.hf_repo_id,
            private=False  # Không cần thiết lập token vì đã đăng nhập sẵn
        )
        bt.logging.success(f"Model uploaded and made public at {config.hf_repo_id}")
    except Exception as e:
        bt.logging.error(f"Failed to update repository visibility: {e}")

if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    asyncio.run(main(config))
