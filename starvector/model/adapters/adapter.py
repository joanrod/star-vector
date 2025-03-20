import torch.nn as nn
import torch.nn.init as init
import torch

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Adapter(nn.Module):
    def __init__(self, input_size, output_size,  adapter_norm="layer_norm", init_type="glorot",  query_length=32, dropout_prob=0.1):
        super().__init__()
        self.query_length = query_length
        self.dropout_prob = dropout_prob
        self.adapter_norm = adapter_norm

        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        self.c_fc = nn.Linear(input_size, input_size*2)
        self.act = Swish()
        self.c_proj = nn.Linear(input_size*2, output_size)
        
        if adapter_norm == "layer_norm":
            self.norm = nn.LayerNorm([self.query_length, output_size])
        elif adapter_norm == "batch_norm":
            self.norm = nn.BatchNorm1d(self.query_length)

        self.init_type = init_type.lower()
        self._initialize_weights()

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == "glorot":
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif self.init_type == "normal":
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                else:
                    raise ValueError("Invalid initialization type specified.")
