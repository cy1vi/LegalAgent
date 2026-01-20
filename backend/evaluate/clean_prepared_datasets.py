import json
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from collections import defaultdict
from config import CleanConfig

# === 1. 模型定义 ===
class FactAccusationClassifier(nn.Module):
    def __init__(self, num_classes, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :] 
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)
        return logits

# === 2. 数据加载 (核心修改：打平数据) ===
class FlattenedDataset(Dataset):
    def __init__(self, flat_data, tokenizer, max_len):
        """
        flat_data: list of dict
        {
            'group_id': int,      # 属于第几行数据
            'type': str,          # 'query' or 'pos'
            'data': dict,         # 原始数据对象 {fact, meta}
            'original_idx': int   # 在 positives 列表中的索引 (如果是 query 则为 -1)
        }
        """
        self.data = flat_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['data'].get('fact', "")
        return {
            'text': text,
            'meta_info': item # 传递元数据以便后续组装
        }

def collate_fn(batch, tokenizer, max_len):
    texts = [b['text'] for b in batch]
    meta_infos = [b['meta_info'] for b in batch]
    
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'meta_infos': meta_infos
    }

# === 3. 核心清洗逻辑 ===
def clean_data():
    cfg = CleanConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 加载资源 ---
    print("正在加载罪名映射...")
    acc_to_idx = {}
    idx_to_acc = {}
    if not os.path.exists(cfg.ACCUSATION_MAP_PATH):
        print(f"错误：找不到文件 {cfg.ACCUSATION_MAP_PATH}")
        return
    with open(cfg.ACCUSATION_MAP_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            acc = line.strip()
            if acc:
                acc_to_idx[acc] = i
                idx_to_acc[i] = acc
    num_classes = len(acc_to_idx)

    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.LAWFORMER_PATH)
    model = FactAccusationClassifier(num_classes, cfg.LAWFORMER_PATH)
    
    if os.path.exists(cfg.MODEL_PATH):
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("找不到模型权重")
        return
    model.to(device)
    model.eval()

    # --- 读取并打平数据 ---
    print(f"正在读取并打平数据: {cfg.INPUT_FILE}")
    flat_data = []
    raw_groups = {} # 存储原始的大组结构，用于最后组装
    
    with open(cfg.INPUT_FILE, 'r', encoding='utf-8') as f:
        for group_idx, line in enumerate(f):
            if not line.strip(): continue
            group_item = json.loads(line)
            raw_groups[group_idx] = group_item # 暂存原始结构
            
            # 1. 添加 Query
            flat_data.append({
                'group_id': group_idx,
                'type': 'query',
                'data': group_item['query'],
                'original_idx': -1
            })
            
            # 2. 添加 Positives
            for pos_idx, pos_item in enumerate(group_item.get('positives', [])):
                flat_data.append({
                    'group_id': group_idx,
                    'type': 'pos',
                    'data': pos_item,
                    'original_idx': pos_idx
                })
    
    print(f"原始组数: {len(raw_groups)}")
    print(f"打平后总条目数: {len(flat_data)} (Query + Positives)")

    dataset = FlattenedDataset(flat_data, tokenizer, cfg.MAX_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        collate_fn=lambda x: collate_fn(x, tokenizer, cfg.MAX_LEN),
        shuffle=False
    )

    # --- 推理与标记 ---
    dirty_flags = defaultdict(lambda: {'query_dirty': False, 'dirty_pos_indices': set(), 'logs': []})
    
    stats = {"total": 0, "discarded": 0}

    print("开始推理与清洗...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            meta_infos = batch['meta_infos']
            
            logits = model(input_ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy() 
            
            for i, meta in enumerate(meta_infos):
                stats["total"] += 1
                current_probs = probs[i]
                item_data = meta['data']
                
                true_accs = set(item_data.get('meta', {}).get('accusation', []))
                
                pred_indices = np.where(current_probs > cfg.PRED_THRESHOLD)[0]
                pred_accs = set([idx_to_acc[idx] for idx in pred_indices])
                
                max_conf = np.max(current_probs)
                max_conf_idx = np.argmax(current_probs)
                max_conf_acc = idx_to_acc[max_conf_idx]

                # === 判定逻辑 ===
                is_clean = True
                
                # 1. 如果有交集 -> Clean
                if not true_accs.isdisjoint(pred_accs):
                    is_clean = True
                # 2. 无交集 且 高置信度 -> Dirty
                elif max_conf > cfg.HIGH_CONFIDENCE_THRESHOLD:
                    is_clean = False
                    stats["discarded"] += 1
                    
                    # 记录日志
                    log_entry = {
                        "type": meta['type'],
                        "idx": meta['original_idx'],
                        "reason": "High Confidence Mismatch",
                        "true": list(true_accs),
                        "pred": list(pred_accs),
                        "top_conf": float(max_conf)
                    }
                    dirty_flags[meta['group_id']]['logs'].append(log_entry)
                # 3. 无交集 但 低置信度 -> Clean (存疑保留)
                else:
                    is_clean = True

                # === 记录结果 ===
                if not is_clean:
                    if meta['type'] == 'query':
                        dirty_flags[meta['group_id']]['query_dirty'] = True
                    else:
                        dirty_flags[meta['group_id']]['dirty_pos_indices'].add(meta['original_idx'])

    # --- 组装与保存 ---
    print("正在重组数据...")
    final_clean_data = []
    final_dirty_data = [] # 这里存被剔除的详细信息

    for group_id in sorted(raw_groups.keys()):
        group = raw_groups[group_id]
        flags = dirty_flags[group_id]
        
        # 1. 如果 Query 本身脏了，整组丢弃（或者你可以选择只丢弃 Query，但这通常意味着这个测试用例废了）
        if flags['query_dirty']:
            group['clean_log'] = flags['logs']
            final_dirty_data.append(group)
            continue
            
        # 2. 过滤 Positives
        original_positives = group.get('positives', [])
        clean_positives = []
        dirty_positives_log = []
        
        for idx, pos_item in enumerate(original_positives):
            if idx in flags['dirty_pos_indices']:
                # 这是一个脏的 positive
                dirty_positives_log.append(pos_item)
            else:
                clean_positives.append(pos_item)
        
        # 更新 group 的 positives
        group['positives'] = clean_positives
        group['positives_count'] = len(clean_positives)
        
        final_clean_data.append(group)
        
        # 如果有被剔除的 positive，也可以记录到 dirty 文件里方便查看
        if dirty_positives_log:
            # 创建一个只包含脏数据的记录用于 debug
            final_dirty_data.append({
                "query": group['query'],
                "dirty_positives": dirty_positives_log,
                "logs": flags['logs']
            })

    print("\n=== 清洗完成 ===")
    print(f"检查总条目数: {stats['total']}")
    print(f"剔除条目数: {stats['discarded']}")
    print(f"最终保留组数: {len(final_clean_data)}")
    
    print(f"正在保存清洗后数据到: {cfg.OUTPUT_CLEAN_FILE}")
    with open(cfg.OUTPUT_CLEAN_FILE, 'w', encoding='utf-8') as f:
        for item in final_clean_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"正在保存剔除日志到: {cfg.OUTPUT_DIRTY_FILE}")
    with open(cfg.OUTPUT_DIRTY_FILE, 'w', encoding='utf-8') as f:
        for item in final_dirty_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    clean_data()