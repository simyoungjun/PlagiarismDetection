import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json
import re

class PDdata(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, args, max_len):
        self.tokenizer = tokenizer
        with open(os.path.join('./data', type_path + ".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.sep_token = tokenizer.sep_token
        self.cls_token = tokenizer.cls_token
        self.pad_token = tokenizer.pad_token
        self.abstracts = []
        self.targets = []
        self.masks = []
        for pair in tqdm(data):
            abstract1 = re.sub(r"[^a-zA-Z0-9]", " ", pair['content1'])
            abstract2 = re.sub(r"[^a-zA-Z0-9]", " ", pair['content2'])
            label = pair['is_plagiarism']
            abstract = self.cls_token + ' ' + abstract1 + ' ' + self.sep_token + ' ' + abstract2 + ' ' + self.sep_token
            tokenized_abstract = self.tokenizer.tokenize(abstract)
            abs_ids = self.tokenizer.encode_plus(tokenized_abstract, max_length=max_len, pad_to_max_length=True, truncation=True)
            tokenized_abstract = abs_ids['input_ids']
            mask = abs_ids['attention_mask']
            self.masks.append(mask)
            self.abstracts.append(tokenized_abstract)
            self.targets.append(label)
            
    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        source_ids = self.abstracts[idx]
        target_ids = self.targets[idx]
        mask = self.masks[idx]

        return source_ids, target_ids, mask
    
def get_dataset(tokenizer, type_path, args):
    return PDdata(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, args=args, max_len=args.max_len)
    