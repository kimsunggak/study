from transformers import AutoTokenizer

class Parameters:
    # model 파라미터
    d_model = 512
    n_head = 8
    num_layers = 6
    dim_ffn = 2048
    num_classes = 2 # 영화 리뷰 긍정/부정 이진 분류
    dropout = 0.1

    tokenizer_model = "bert-base-uncased"
    batch_size = 32
    max_len = 512

    @classmethod
    def get_vocab_size(cls):
        tokenizer = AutoTokenizer.from_pretrained(cls.tokenizer_model)
        return tokenizer.vocab_size
    
    # 정적 속성으로도 접근 가능하도록
    vocab_size = 30522  # bert-base-uncased의 vocab_size