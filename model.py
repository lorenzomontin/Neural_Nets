import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    """Feed-forward neural network for tabular inputs."""

    def __init__(self, input_features, hidden_dim, output_features, dropout):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_features # how many features are entering the next layer

        # Hidden layer

        for h_layer in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_layer)) # (features in, features out)
            layers.append(nn.ReLU()) 
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_layer

        # Output layer
        layers.append(nn.Linear(prev_dim, output_features)) 
        self.net = nn.Sequential(*layers) # chaining togehter layers, the * unpacks python lists as Sequential() wants

    def forward(self, x):
        return self.net(x) # also this is not clear, though I know here we are pushing activations through the MLP archietcutre


# Smoke test: Code inside this block only runs when you execute python model.py
#  directly. it is skipped when another file like train.py does from model import TabularMLP.
if __name__ == "__main__":
    model = TabularMLP(
        input_features= 30, hidden_dim = (64, 32), output_features=1,dropout=0.2 
    )
    dummy_x = torch.randn(8,30)
    output = model(dummy_x)
    print("Input shape:",dummy_x.shape)
    print("Output shape", output.shape)
