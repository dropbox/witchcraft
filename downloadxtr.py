from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/xtr-base-multilingual", local_dir="xtr-base-multilingual", local_dir_use_symlinks=False, revision="main")
