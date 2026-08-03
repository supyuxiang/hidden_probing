import torch
import torch.nn as nn


class Classifier_Linear(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int=1,
        is_half:bool=False,
    ):
        super(Classifier_Linear,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_half=is_half

        self.linear = nn.Linear(self.input_dim,self.output_dim)
        # self.pre_norm = nn.LayerNorm(self.input_dim)

        self._init_weight()
    
    def _init_weight(self):
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.is_half:
            self.linear.to(torch.bfloat16)

    def forward(self,x:torch.Tensor):
        # x: batch_size, input_dim -> logits: batch_size, output_dim
        # x = self.pre_norm(x)
        return self.linear(x)




class Classifier_MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, 1)
        # self.pre_norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim), # add layer norm here
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.Dropout(dropout_prob),
                )
                for _ in range(num_layers)
            ]
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) -> logit: (batch, 1)
        # x = self.pre_norm(x)
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x) + x
        return self.proj_out(x)



if __name__ == '__main__':
    model = Classifier_Linear(
        input_dim=2048,
        output_dim=2
    )
    data = torch.randn(1000,2048)

    with torch.inference_mode():
        o = model(data)
    print(o.shape)
    o_choose = torch.argmax(o,dim=-1)
    print(o_choose)

    
