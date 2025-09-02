from src import DataModule, EncoderOnlyModel,Parameters,Trainer
from datasets import load_dataset
import torch
import torch.nn as nn

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("stanfordnlp/imdb")
    train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    test_data = dataset['test']

    data = DataModule(train_data, val_data, test_data)
    train_dataloader = data.train_dataloader()
    test_dataloader = data.test_dataloader()
    val_dataloader = data.val_dataloader()

    encoder = EncoderOnlyModel(
        vocab_size=Parameters.vocab_size,
        d_model=Parameters.d_model,
        n_head=Parameters.n_head,
        num_layers=Parameters.num_layers,
        dim_ffn=Parameters.dim_ffn,
        num_classes=Parameters.num_classes
    )

    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)

    trainer = Trainer(encoder,device)

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train(train_dataloader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 검증
        val_loss, val_acc = trainer.evaluate(val_dataloader, criterion)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 테스트
    print("\n=== Testing ===")
    test_loss, test_acc = trainer.evaluate(test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    run()
