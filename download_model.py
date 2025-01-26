from huggingface_hub import snapshot_download

model_name = "Diginsa/Plant-Disease-Detection-Project"
local_dir = "model/plant_disease_model"

# Download the model and save it locally
snapshot_download(repo_id=model_name, local_dir=local_dir)

print("Model downloaded successfully!")
