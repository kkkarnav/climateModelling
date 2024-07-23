import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from pb_loss import *
from tcnn import TempCNN
from tempdataset import TemperatureDataset

def train(model, train_loader, criterion, optimizer, writer, epoch):
    model.train()
    train_loss = 0
    for sequences, labels, time, constants in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences, constants, time)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    return train_loss

def validate(model, val_loader, criterion, writer, epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sequences, labels, time, constants in val_loader:
            outputs = model(sequences, constants, time)
            loss = criterion(outputs, labels.squeeze())
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    writer.add_scalar('Loss/val', val_loss, epoch)
    return val_loss

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_outputs = []
    all_labels = []
    all_times = []
    with torch.no_grad():
        for sequences, labels, time, constants in test_loader:
            outputs = model(sequences, constants, time)
            loss = criterion(outputs, labels.squeeze())
            test_loss += loss.item()
            all_outputs.append(outputs.numpy())
            all_labels.append(labels.numpy())
            all_times.append(time.numpy())
    
    test_loss /= len(test_loader)
    return test_loss, np.concatenate(all_outputs, axis=0), np.concatenate(all_labels, axis=0), np.concatenate(all_times, axis=0)

def plot_results(all_outputs, all_labels, all_times, label_length, plot_dir, name=None, filename=None):
    years = np.unique(all_times[:, 3])
    fig, axes = plt.subplots(len(years), 1, figsize=(12, 6 * len(years)), sharex=True)
    if len(years) == 1:
        axes = [axes]
    
    colors = plt.cm.Blues(np.linspace(0.3, 1, label_length))
    
    for i, year in enumerate(years):
        print(i, year)
        year_indices = (all_times[:, 3] == year)
        outputs = all_outputs[year_indices]
        labels = all_labels[year_indices]
        
        day_labels = labels[:, 0]
        num_points = len(day_labels)
        axes[i].plot(np.arange(num_points), day_labels.flatten(), label=f'Actual', color = 'orange')
        for day in range(label_length):
            
            day_outputs = outputs[:, day]
            
            x = np.arange(day, num_points + day)
            
            axes[i].plot(x, day_outputs.flatten(), label=f'Predicted Day {day+1}', color = colors[day])
        
        axes[i].legend()
        axes[i].set_title(f'Year {int(year)}')
    
    plt.xlabel('Time')
    plt.tight_layout()
    if name:
        plt.savefig(os.path.join(plot_dir, f"{name['name']}_{name['lat']}_{name['lon']}.png"))
    elif filename:
        plt.savefig(os.path.join(plot_dir, f'{filename}.png'))
    else:
        plt.savefig(os.path.join(plot_dir, 'predicted_vs_actual.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate TempCNN model.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--data_col', type=str, required=True, help='Comma-separated names of data columns.')
    parser.add_argument('--constant_cols', type=str, default='', help='Comma-separated names of constant columns (if any).')
    parser.add_argument('--label_col', type=str, required=True, help='Comma-separated names of label columns.')
    parser.add_argument('--seq_length', type=int, default=30, help='Length of the input sequences.')
    parser.add_argument('--label_length', type=int, default=7, help='Length of the output sequences (labels).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs.')
    parser.add_argument('--model_save_path', type=str, default='model.pth', help='Path to save the trained model.')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save the plots.')
    parser.add_argument('--plot', type=bool, default=True, help="Whether or not to plot results")
    parser.add_argument('--train', type=bool, default=True, help="Train Model")
    parser.add_argument('--test', type=bool, default=True, help="Test Model")
    parser.add_argument('--no_train', dest='train', action='store_false')
    parser.add_argument('--no_test', dest='test', action='store_false')
    parser.add_argument('--no_plot', dest='plot', action='store_false')
    parser.add_argument('--model_path', type=str, default=None, help="Path to pretrained model")
    parser.add_argument('--multiple_grids', type=bool, default=False, help="Flag to indicate multiple grids")
    parser.add_argument('--grid_coords', type=str, default='', help='Comma-separated list of grid coordinates in the format lat1,lon1,name1;lat2,lon2,name2;...')
    parser.add_argument('--peak_bias', action='store_true', help='Use peak biased loss function', default=False)
    parser.add_argument('--checkpoints', action='store_true', help='Save Checkpoints', default=False)
    parser.add_argument('--save_dir', type=str, help='Save directory', default="./checkpoints")
    parser.add_argument('--save_interval', type=int, help='Number of Epochs to Save after', default="10")
    parser.add_argument('--desc', type=str, default=None, help="Experiment Description")
    args = parser.parse_args()
    
    print(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.desc:
        log_dir = os.path.join(args.log_dir, f'{args.desc}_run_{timestamp}')
    else:
        log_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    
    data = pd.read_csv(args.data_file)
    # data = data[data['tmax'] < 53]
    datacol = args.data_col.split(',')
    constantcols = args.constant_cols.split(',') if args.constant_cols else []
    labelcol = args.label_col.split(',')
    
    num_constants = len(constantcols)
    model = TempCNN(seq_length=args.seq_length, num_constants=num_constants, label_length=args.label_length)
    
    if args.peak_bias:
        criterion = PeakBiasedMAE()
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir=log_dir)

    if args.train:
        print("Training")
        train_dataset = TemperatureDataset(data, datacol, labelcol, constantcols=constantcols, split='train')
        val_dataset = TemperatureDataset(data, datacol, labelcol, constantcols=constantcols, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, criterion, optimizer, writer, epoch)
            val_loss = validate(model, val_loader, criterion, writer, epoch)
            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), f"{args.save_dir}/epoch_{epoch}.pt")
            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')
        torch.save(model.state_dict(), args.model_save_path)
    
    elif (not args.train) and args.model_path:
        state = torch.load(args.model_path)
        model.load_state_dict(state)
        print("Loaded")

    else:
        print("Neither training a new model not loading a trained model, exiting")
        exit()
    
    if args.test:
        print("Testing")
        test_dataset = TemperatureDataset(data, datacol, labelcol, constantcols=constantcols, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_loss, all_outputs, all_labels, all_times = test(model, test_loader, criterion)
        writer.add_scalar('Loss/test', test_loss)
        print(f'Test Loss: {test_loss}')
    
    writer.close()

    
    if args.plot:
        if args.multiple_grids and 'lat' in constantcols and 'lon' in constantcols:
            print("Plotting Multiple Grids")
            grids = args.grid_coords.split(';')
            for grid in grids:
                name, lat, lon = grid.split(',')
                latdata = data[data['lat'] == float(lat)]
                filtered_data = latdata[latdata['lon'] == float(lon)]
                if filtered_data.empty:
                    print(f"No data for {name}, at coordinates {(lat, lon)}")
                    continue
                test_dataset = TemperatureDataset(filtered_data, datacol, labelcol, constantcols=constantcols, split='test')
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                test_loss, all_outputs, all_labels, all_times = test(model, test_loader, criterion)
                print(f'Test Loss for {name} ({lat}, {lon}): {test_loss}')
                
                plot_results(all_outputs, all_labels, all_times, args.label_length, args.plot_dir, name={"lat": lat, "lon": lon, "name": name})
        elif args.multiple_grids and ('lat' not in constantcols or 'lon' not in constantcols):
            print("Lat and Lon not constant across series, cannot plot grid-wise, exiting")
            exit()
        else:
            print("Plotting Single Grid")
            test_dataset = TemperatureDataset(data, datacol, labelcol, constantcols=constantcols, split='test')
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            test_loss, all_outputs, all_labels, all_times = test(model, test_loader, criterion)            
            plot_results(all_outputs, all_labels, all_times, args.label_length, args.plot_dir, {"lat": "all", "lon": "all", "name": "all"})
        
if __name__ == "__main__":
    main()