import torch
import torch.nn as nn
import torch.nn.functional as F

class TempCNN(nn.Module):
    def __init__(self, seq_length, num_constants, label_length):
        super(TempCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        conv_output_size = 32 * (seq_length // 2 // 2)  
        self.fc_conv = nn.Linear(conv_output_size, 64)
        self.fc_constants = None
        self.constants = False
        if num_constants > 0:
            self.fc_constants = nn.Linear(num_constants, 16)
            self.constants = True
        self.fc_time = nn.Linear(6, 16)  # Start and end time combined have 6 features (year, month, day)
        
        input_size = 64 + (16 if num_constants > 0 else 0) + 16
        self.fc_combined = nn.Linear(input_size, label_length)
        
    def forward(self, x, constants, time):
        x = x.squeeze()
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_conv(x))
        if self.constants:
            constants = F.relu(self.fc_constants(constants))
        
        # time = torch.cat((start_time, end_time), dim=1)
        time = F.relu(self.fc_time(time))
        
        if constants.nelement() != 0:
            combined = torch.cat((x, constants, time), dim=1)
        else:
            combined = torch.cat((x, time), dim=1)
        
        out = self.fc_combined(combined)
        return out
