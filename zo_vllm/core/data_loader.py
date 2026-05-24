"""
SST2 Dataset Loader - 对齐 LOZO large_models/tasks.py + utils.py + templates.py

数据流程：
1. load_dataset('glue', 'sst2')
2. build_sample: {"sentence": text, "label": 0/1}
3. encode_prompt_for_training: teacher forcing prompt + tokenization
"""

import os
import random
import numpy as np
from typing import List, Dict, Optional
from datasets import load_dataset


class SST2Template:
    """
    对齐 templates.py:27-41 SST2Template
    """
    verbalizer = {0: "terrible", 1: "great"}
    
    def encode(self, sample: Dict) -> str:
        """Prompt without answer"""
        text = sample["sentence"].strip()
        return f"{text} It was"
    
    def verbalize(self, sample: Dict, candidate: int) -> str:
        """Prompt with answer"""
        text = sample["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"


class SST2Dataset:
    """
    对齐 tasks.py:107-128 SST2Dataset
    
    SST2 是 binary classification：
    - label=0: negative (verbalize: "terrible")
    - label=1: positive (verbalize: "great")
    """
    
    train_sep = "\n\n"
    template = SST2Template()
    
    def __init__(
        self,
        tokenizer,
        num_train: int = 1000,
        num_dev: int = 500,
        num_eval: int = 1000,
        seed: int = 0,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.num_train = num_train
        self.num_dev = num_dev
        self.num_eval = num_eval
        self.seed = seed
        self.max_length = max_length
        
        # 对齐 run_lozo.py:179: OPT tokenizer bos_token_id=0
        if hasattr(tokenizer, 'bos_token_id'):
            tokenizer.bos_token_id = 0
        
        self.load_dataset()
        self.sample_train_sets()
    
    def load_dataset(self):
        """对齐 tasks.py:112-120"""
        d = load_dataset('glue', 'sst2')
        
        # Build samples
        train_samples = []
        for example in d["train"]:
            train_samples.append(self.build_sample(example))
        
        valid_samples = []
        for example in d["validation"]:
            valid_samples.append(self.build_sample(example))
        
        self.samples = {
            "train": train_samples,
            "valid": valid_samples,
        }
    
    def build_sample(self, example: Dict) -> Dict:
        """对齐 tasks.py:123-125"""
        label = int(example["label"])
        return {
            "id": example["idx"],
            "sentence": example["sentence"],
            "label": label,
            "candidates": [0, 1],
        }
    
    def sample_train_sets(self):
        """对齐 tasks.py:60-89"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Sample train + dev
        train_samples = self.samples["train"]
        if self.num_train + self.num_dev <= len(train_samples):
            indices = np.random.permutation(len(train_samples)).tolist()
            indices = indices[:self.num_train + self.num_dev]
        else:
            indices = list(range(len(train_samples)))
        
        self.train_data = [train_samples[i] for i in indices[:self.num_train]]
        self.dev_data = [train_samples[i] for i in indices[self.num_train:self.num_train + self.num_dev]]
        
        # Sample eval
        valid_samples = self.samples["valid"]
        if self.num_eval <= len(valid_samples):
            indices = np.random.permutation(len(valid_samples)).tolist()[:self.num_eval]
        else:
            indices = list(range(len(valid_samples)))
        
        self.eval_data = [valid_samples[i] for i in indices]
    
    def encode_prompt_for_training(self, sample: Dict) -> Dict:
        """
        Teacher forcing 模式
        
        对齐 run_lozo.py:389-391 + utils.py:105-172
        
        Returns:
            {
                "input_ids": list[int],
                "labels": list[int],
                "option_len": int,  # 只计算最后 option_len 个 token 的 loss
            }
        """
        # Verbalize (teacher forcing)
        prompt_text = self.template.verbalize(sample, sample["label"])
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt_text)
        
        # Truncate if needed (left truncate)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
        
        # Labels: teacher forcing (只计算 option 部分)
        labels = input_ids.copy()
        
        # option_len: verbalizer 只有 1 个 token
        # "terrible" 或 "great" 通常对应 1 个 token
        option_len = 1
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "option_len": option_len,
        }
    
    def encode_prompt_for_loss(self, sample: Dict) -> Dict:
        """
        用于 loss 计算（teacher forcing）
        
        对齐 utils.py:134-146 + run_lozo.py:389-391
        """
        return self.encode_prompt_for_training(sample)
    
    def get_batch(self, data: List[Dict], batch_size: int = 16, start_idx: int = 0) -> List[Dict]:
        """获取一个 batch"""
        end_idx = min(start_idx + batch_size, len(data))
        batch = []
        for i in range(start_idx, end_idx):
            encoded = self.encode_prompt_for_training(data[i])
            batch.append(encoded)
        return batch
    
    def __len__(self):
        return len(self.train_data)


def test_sst2_dataset():
    """测试 SST2 数据加载"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)
    tokenizer.bos_token_id = 0
    
    dataset = SST2Dataset(
        tokenizer,
        num_train=10,
        num_dev=5,
        num_eval=5,
        seed=0,
    )
    
    print(f"Train samples: {len(dataset.train_data)}")
    print(f"Dev samples: {len(dataset.dev_data)}")
    print(f"Eval samples: {len(dataset.eval_data)}")
    
    # 测试 encode
    sample = dataset.train_data[0]
    encoded = dataset.encode_prompt_for_training(sample)
    
    print(f"\nSample 0:")
    print(f"  Sentence: {sample['sentence']}")
    print(f"  Label: {sample['label']}")
    print(f"  Prompt: {dataset.template.verbalize(sample, sample['label'])}")
    print(f"  Input IDs length: {len(encoded['input_ids'])}")
    print(f"  Option len: {encoded['option_len']}")
    
    # 测试 batch
    batch = dataset.get_batch(dataset.train_data, batch_size=2)
    print(f"\nBatch size: {len(batch)}")


if __name__ == "__main__":
    test_sst2_dataset()