"""
A SIMPLE CONV1D Neural Network same as Deezer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, detect_encoder=False):
        super().__init__()
        self.detect_encoder = detect_encoder

        self.conv1 = nn.Conv1d(input_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(1)

        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(3)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(3)

        self.conv4 = nn.Conv1d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(3)

        self.conv5 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(3)

        self.conv6 = nn.Conv1d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.pool6 = nn.MaxPool1d(3)

        self.dropout = nn.Dropout(0.2)
        self.head_dense = nn.Linear(512, 64)
        self.fc_out = nn.Linear(64, 1)
        if detect_encoder:
            self.fc_encoder = nn.Linear(64, detect_encoder)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.head_dense(x))
        x = self.dropout(x)

        out1 = torch.sigmoid(self.fc_out(x))
        if self.detect_encoder:
            out2 = F.softmax(self.fc_encoder(x), dim=-1)
            return out1, out2
        return out1
