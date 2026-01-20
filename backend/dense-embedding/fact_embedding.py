import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel 
from config import Config



class LegalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, model_type="bge-m3"):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        fact = item['fact']
        articles = item['meta']['relevant_articles']
        # 编码fact文本
        if self.model_type.lower() == "bge-m3":
            # BGE-M3 不需要tokenization，直接返回原始文本
            return {
                'fact': fact,
                'articles': articles
            }
        else:
            # Lawformer 需要tokenization
            encoding = self.tokenizer(
                fact,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'articles': articles
            }

def custom_collate_fn(batch, model_type="bge-m3"):
    if model_type.lower() == "bge-m3":
        # BGE-M3不需要tokenization，直接返回文本列表
        facts = [item['fact'] for item in batch]
        articles = [item['articles'] for item in batch]
        return {
            'facts': facts,
            'articles': articles
        }
    else:
        # Lawformer需要tokenized tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        articles = [item['articles'] for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'articles': articles
        }

# 获取fact的embedding
def get_fact_embeddings(data_loader, model, device, model_type="lawformer"):
    if model_type.lower() == "bge-m3":
        embeddings = []
        article_lists = []

        for batch in tqdm(data_loader, desc="Processing batches"):
            facts = batch['facts']
            articles = batch['articles']

            batch_embeddings = model.encode(
                facts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )

            dense_vecs = batch_embeddings["dense_vecs"]
            embeddings.append(dense_vecs)
            article_lists.extend(articles)

        return np.vstack(embeddings), article_lists
    else:
        model.eval()
        embeddings = []
        article_lists = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                articles = batch['articles']     
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)
                article_lists.extend(articles)
        return np.vstack(embeddings), article_lists



# 主函数
def main():
    parser = argparse.ArgumentParser(description="Generate fact embeddings using Lawformer")
    parser.add_argument("--train_data_path", type=str, default=Config.RAW_DATA_PATH, help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default=r"F:\LegalAgent\backend\data\fact_embeddings", help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--model_type", type=str, choices=["lawformer", "bge-m3"], default="bge-m3", help="Type of embedding model to use")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 根据模型类型加载模型
    if args.model_type.lower() == "bge-m3":
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
        tokenizer = None
    else:
        LAWFARMER_MODEL_NAME = "thunlp/Lawformer"
        tokenizer = AutoTokenizer.from_pretrained(LAWFARMER_MODEL_NAME)
        model = AutoModel.from_pretrained(LAWFARMER_MODEL_NAME).to(device)


    # 创建数据集和数据加载器，使用自定义的collate函数
    dataset = LegalDataset(args.train_data_path, tokenizer, model_type=args.model_type)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=lambda batch: custom_collate_fn(batch, args.model_type))
    
    # 获取fact embeddings
    fact_embeddings, article_lists = get_fact_embeddings(data_loader, model, device, args.model_type)
    
    # 保存embeddings
    embedding_output_path = os.path.join(args.output_dir, f"fact_embeddings_{args.model_type}.npy")
    article_output_path = os.path.join(args.output_dir, f"article_lists_{args.model_type}.json")
    
    np.save(embedding_output_path, fact_embeddings)
    with open(article_output_path, "w", encoding="utf-8") as f:
        json.dump(article_lists, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(fact_embeddings)} fact embeddings with dimension {fact_embeddings.shape[1]}")
    print(f"Embeddings saved to: {embedding_output_path}")
    print(f"Article lists saved to: {article_output_path}")

if __name__ == "__main__":
    main()