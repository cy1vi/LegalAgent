import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_path = "BAAI/bge-reranker-v2-m3"
print('device cuda available:', torch.cuda.is_available())
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True,device_map="cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, trust_remote_code=True,device_map="cpu")
    print('model dtype:', model.dtype)
    model.eval()
    print('model loaded successfully')
except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
