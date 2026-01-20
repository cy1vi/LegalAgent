import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm
import config

# 加载Lawformer模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_path)

# 自定义数据集类
class LegalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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

# 自定义collate函数，处理不同长度的articles
def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    articles = [item['articles'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'articles': articles
    }

# 获取fact的embedding
def get_fact_embeddings(data_loader, model, device):
    model.eval()
    embeddings = []
    article_lists = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            # 将数据移动到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            articles = batch['articles']     
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 使用[CLS] token的表示作为整个句子的embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
            article_lists.extend(articles)
    return np.vstack(embeddings), article_lists

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Generate fact embeddings using Lawformer")
    parser.add_argument("--train_data_path", type=str, default=config.train_data_path, help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default=config.output_dir, help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size for processing")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型并移动到设备
    model = AutoModel.from_pretrained(config.model_path).to(device)

    # 创建数据集和数据加载器，使用自定义的collate函数
    dataset = LegalDataset(args.train_data_path, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # 获取fact embeddings
    fact_embeddings, article_lists = get_fact_embeddings(data_loader, model, device)
    
    # 保存embeddings
    embedding_output_path = os.path.join(args.output_dir, "fact_embeddings.npy")
    article_output_path = os.path.join(args.output_dir, "article_lists.json")
    
    np.save(embedding_output_path, fact_embeddings)
    with open(article_output_path, "w", encoding="utf-8") as f:
        json.dump(article_lists, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(fact_embeddings)} fact embeddings with dimension {fact_embeddings.shape[1]}")
    print(f"Embeddings saved to: {embedding_output_path}")
    print(f"Article lists saved to: {article_output_path}")

if __name__ == "__main__":
    main()