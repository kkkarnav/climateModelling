import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TemperatureDataset(Dataset):
    def __init__(self, data, datacol, labelcol, constantcols=None, split='train', seq_length=15, label_length=3, train_years=(1951, 2008), val_years=(2009, 2013), test_years=(2014, 2023)):
        self.data = data
        self.datacol = datacol
        self.labelcol = labelcol
        self.constantcols = constantcols if constantcols else []
        self.split = split
        self.seq_length = seq_length
        self.label_length = label_length
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.split_data()
        self.prepare_sequences()

    def split_data(self):
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['year'] = self.data['time'].dt.year        
        if self.split == 'train':
            self.data = self.data[(self.data['year'] >= self.train_years[0]) & (self.data['year'] <= self.train_years[1])]
        elif self.split == 'val':
            self.data = self.data[(self.data['year'] >= self.val_years[0]) & (self.data['year'] <= self.val_years[1])]
        elif self.split == 'test':
            self.data = self.data[(self.data['year'] >= self.test_years[0]) & (self.data['year'] <= self.test_years[1])]
        
    def prepare_sequences(self):
        grouped = self.data.groupby(self.constantcols) if self.constantcols else [("", self.data)]
        
        self.sequences = []
        self.labels = []
        self.metadata = []
        for _, group in grouped:
            group = group.sort_values(by='time')
            data_points = group[self.datacol].values
            labels = group[self.labelcol].values
            constants = group[self.constantcols].iloc[0].values if self.constantcols else []
            
            for i in range(len(data_points) - self.seq_length - self.label_length + 1):
                seq = data_points[i:i+self.seq_length]
                label = labels[i+self.seq_length:i+self.seq_length+self.label_length]
                start_time = group['time'].iloc[i]
                end_time = group['time'].iloc[i + self.seq_length - 1]
                self.sequences.append(seq)
                self.labels.append(label)
                self.metadata.append((constants, start_time, end_time))
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        constants, start_time, end_time = self.metadata[idx]
        constants = torch.tensor(constants, dtype=torch.float32) if self.constantcols else torch.tensor([])
        time = torch.tensor([start_time.year, start_time.month, start_time.day, end_time.year, end_time.month, end_time.day], dtype=torch.float32)
        return sequence, label, time, constants
    