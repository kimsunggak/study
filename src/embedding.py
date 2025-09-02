import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # (max_len, d_model) 형태로 초기화
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        # 위치마다 고유한 값 부여
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
        # 짝수 인덱스에는 사인 함수, 홀수 인덱스에는 코사인 함수 적용
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model) 형태로 변경 - 배치 추가
        self.register_buffer('pe', pe) # 학습 안하게 등록

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class InputEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,dropout):
        super().__init__()
        # 사전학습된 임베딩 모델로 변경 해보기
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len)
        # 랜덤하게 일부 단어를 지움으로써 과적합을 방지함
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.token_embed(x)
        x = self.pos_embed(x)
        x = self.dropout(x)
        return x