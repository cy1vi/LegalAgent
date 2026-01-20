import os

class Config:
    project_path = "F:/LegalAgent" 
    train_data_path = os.path.join(project_path, "dataset/final_all_data/first_stage/train.json")
    output_dir = os.path.join(project_path, "backend/data/fact_embeddings")
    model_path = "F:/LegalAgent/model/Lawformer"

    batch_size = 8
    