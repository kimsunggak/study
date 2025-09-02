import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from .config import Parameters
from transformers import AutoTokenizer

class MyDataset(Dataset):
    def __init__(self, data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        processed_text = torch.tensor(self.tokenizer.encode(item['text'],truncation=True,max_length=Parameters.max_len), dtype=torch.int64)
        label = torch.tensor(item['label'], dtype=torch.int64)
        return processed_text, label

class DataModule():
    def __init__(self, train_data, val_data, test_data, batch_size=None):
        self.tokenizer = AutoTokenizer.from_pretrained(Parameters.tokenizer_model)
        self.batch_size = Parameters.batch_size
        
        self.train_dataset = MyDataset(train_data, self.tokenizer)
        self.val_dataset = MyDataset(val_data, self.tokenizer)
        self.test_dataset = MyDataset(test_data, self.tokenizer)

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_text, _label) in batch:
            label_list.append(_label)
            text_list.append(_text)

        labels = torch.tensor(label_list, dtype=torch.int64)
        texts = pad_sequence(text_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return texts, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_batch
        )