import torch
import torch.nn as nn
from .embedding import InputEmbedding
from .config import Parameters

class EncoderOnlyModel(nn.Module):
    def __init__(self,vocab_size,d_model,n_head,num_layers,dim_ffn,num_classes):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model, max_len=1024, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = Parameters.d_model,
            nhead = Parameters.n_head,
            dim_feedforward = Parameters.dim_ffn,
            dropout = Parameters.dropout,
            # 텐서 순서 배치를 처음으로
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, Parameters.num_layers)
        self.classifier = nn.Linear(Parameters.d_model, Parameters.num_classes)


    def forward(self,x,src_key_padding_mask):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls = x[:, 0, :]
        x = self.classifier(cls)
        return x