# download huggingface pretrain model

import os
import sys
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    repo_id = sys.argv[1]
    snapshot_download(
        repo_id = repo_id, 
        cache_dir = './', 
    )
    print("Done")
