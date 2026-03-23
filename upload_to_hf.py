from huggingface_hub import HfApi, login

api = HfApi()

repo_id = "hxixixh/gumbel-distillation-models"  

local_folder = "./checkpoints"

print(f"🚀 准备创建并上传至: {repo_id}")

# 4. 创建云端仓库（如果存在则跳过）
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# 5. 上传整个文件夹
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload weights for ICLR 2025: Gumbel Distillation"
)

print("✅ 上传成功！")