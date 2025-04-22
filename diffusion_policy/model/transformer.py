import torch
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, obs_dim=38, action_dim=10, seq_len=16, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, obs, action_seq):
        # obs: (B, 38), action_seq: (B, 16, 10)
        B = obs.size(0)

        # Project inputs
        obs_emb = self.obs_proj(obs)  # (B, d_model)
        obs_emb = obs_emb.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, 16, d_model)

        action_emb = self.action_proj(action_seq)  # (B, 16, d_model)

        # Combine observation embedding with action sequence embedding
        x = obs_emb + action_emb + self.pos_embedding.unsqueeze(0)  # (B, 16, d_model)

        # Encode
        encoded = self.encoder(x)  # (B, 16, d_model)

        # Decode to (B, 16, 1)
        output = self.decoder(encoded)
        return output