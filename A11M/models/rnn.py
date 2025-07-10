import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # Define param
        self._reset_parameter()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # h = tanh(U⋅X+W⋅h{t−1} +B)
        h = torch.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        return h


    def _reset_parameter(self):
        self.Wxh = nn.Parameter(torch.randn(self._input_size, self._hidden_size))
        self.Whh = nn.Parameter(torch.randn(self._hidden_size, self._hidden_size))
        self.bh  = nn.Parameter(torch.zeros(self._hidden_size))
    

class RNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self._hidden_size = hidden_size
        self._cell = RNNCell(input_size, hidden_size)
    
    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)
        h_0: (batch_size, hidden_size)
        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape  # torch.zeros((2, 3, 4)) -> batch_size = 2, seq_len = 3, feature = 4

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self._hidden_size, device=x.device)
        
        hs = []
        h = h_0

        for t in range(seq_len):
            x_t = x[:, t, :]                            # (batch_size, input_size)
            h: torch.Tensor = self._cell(x_t, h)        # (batch_size, hidden_size)
            hs.append(h.unsqueeze(1))                   # (batch_size, 1, hidden_size)
        
        return torch.cat(hs, dim=1)    # (batch_size, seq_len, hidden_size)

class RNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = RNNLayer(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) - token indices
        returns: logits (batch_size, seq_len, vocab_size)
        """
        embedded: torch.Tensor = self.embedding(x)                   # (batch, seq_len, embed_size)
        rnn_out: torch.Tensor = self.rnn(embedded)                   # (batch, seq_len, hidden_size)
        logits: torch.Tensor = self.fc(rnn_out)                      # (batch, seq_len, vocab_size)
        return logits
    



