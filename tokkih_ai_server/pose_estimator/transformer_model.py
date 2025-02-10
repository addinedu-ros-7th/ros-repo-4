import torch
import torch.nn as nn

class TransformerPoseClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout, max_len):
        super(TransformerPoseClassifier, self).__init__()
        
        # 입력 프로젝션: 2D 좌표를 hidden_dim 차원으로 변환
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding
        self.positional_encoding = nn.Embedding(max_len, hidden_dim)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 최종 분류기
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.input_projection(x) + self.positional_encoding(positions)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 마지막 프레임의 출력 사용
        x = self.dropout(x)
        return self.fc(x)
