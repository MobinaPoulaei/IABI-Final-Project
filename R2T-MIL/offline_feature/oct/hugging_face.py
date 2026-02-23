from huggingface_hub import hf_hub_download

retfound_ckpt_path = hf_hub_download(
    repo_id="YukunZhou/RETFound_mae_natureOCT",
    filename="RETFound_mae_natureOCT.pth",
    local_dir="./weights",
    token="YOUR HF TOKEN"
)
