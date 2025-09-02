import torch
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self, train_dataloader, optimizer, criterion):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_samples = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            texts, labels = batch
            texts = texts.to(self.device)
            labels = labels.to(self.device)

            # 패딩 마스크 생성 (패딩 토큰 위치를 True로)
            pad_token_id = train_dataloader.dataset.tokenizer.pad_token_id
            src_key_padding_mask = (texts == pad_token_id)

            optimizer.zero_grad()

            outputs = self.model(texts, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (outputs.argmax(dim=-1) == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss, total_acc = 0, 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                texts, labels = batch
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                pad_token_id = dataloader.dataset.tokenizer.pad_token_id
                src_key_padding_mask = (texts == pad_token_id)

                outputs = self.model(texts, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                total_acc += (outputs.argmax(dim=-1) == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / total_samples
        return avg_loss, avg_acc