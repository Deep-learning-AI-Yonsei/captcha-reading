import torch
import torch.nn as nn
import timm

class CaptchaDeiT(nn.Module):
    def __init__(self, num_classes, sequence_length):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.base_model = timm.create_model('deit_base_patch16_224', pretrained=True)
        num_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        self.fc = nn.Linear(num_features, num_classes * sequence_length)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        x = x.view(-1, self.sequence_length, self.num_classes)
        return x