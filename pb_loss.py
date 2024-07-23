import torch.nn as nn
import torch
class PeakBiasedMAE(nn.Module):
    def __init__(self):
        super(PeakBiasedMAE, self).__init__()
    
    def forward(self, r, r_hat):
        abs_errors = torch.abs(r_hat - r)
        
        underestimating_mask = (r_hat < r)
        overestimating_mask = (r_hat >= r)
        
        loss_under = underestimating_mask.float() * abs_errors**1.5
        loss_over = overestimating_mask.float() * abs_errors
        
        total_loss = loss_under + loss_over
        return total_loss.mean()
    