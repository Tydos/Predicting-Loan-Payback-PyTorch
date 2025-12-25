import torch.nn as nn

class loan_predictor(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features,64),
            nn.ReLU(),
            nn.Linear(64,52),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(52,28),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(28,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
        
    def forward(self,x):
        return self.model(x);