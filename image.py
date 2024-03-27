import os
import time

from modal import Image, Stub, enter, exit, gpu, method

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

GPU_CONFIG = gpu.A100(memory=40, count=1)

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt"],  # Using safetensors
    )
    move_cache()

llm_image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(
        "transformers",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
        "sentencepiece",
        "simple-parsing",
        "bitsandbytes",
        "scipy",
        "datasets",
        "wandb",
        "tensorboard",
        "xformers",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, timeout=60 * 20)
)

stub = Stub("mistral-7b-v0.2", image=llm_image)

