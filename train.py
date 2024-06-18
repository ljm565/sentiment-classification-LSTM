import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import pickle
from tokenizer import Tokenizer
import numpy as np
import time
import random
import sys

from config import Config
from utils_func import *
from data_utils import DLoader
from model import SentimentLSTM



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # define tokenizer
        self.tokenizer = Tokenizer(self.config, self.data_path['train'])
        self.config.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        elif self.mode == 'test':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = SentimentLSTM(self.config, self.pad_token_id, self.device).to(self.device)
        self.criterion = nn.BCELoss()
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_val_acc = 0 if not self.continuous else self.loss_data['best_val_acc']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        train_acc_history = [] if not self.continuous else self.loss_data['train_acc_history']
        val_acc_history = [] if not self.continuous else self.loss_data['val_acc_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss, total_acc = 0, 0
                for i, (x, y) in enumerate(self.dataloaders[phase]):
                    batch = x.size(0)
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output, _ = self.model(x)
                        loss = self.criterion(output, y)
                        acc = ((output > 0.5).float()==y).float().sum()/batch

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    total_loss += loss.item()*batch
                    total_acc += acc.item()*batch
                    if i % 50 == 0:
                        print('Epoch {}: {}/{} step loss: {}, step acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), acc.item()))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
                print('{} loss: {:4f}, acc: {:4f}\n'.format(phase, epoch_loss, epoch_acc))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                if phase == 'test':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
                    early_stop += 1
                    if epoch_acc > best_val_acc:
                        early_stop = 0
                        best_val_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer)
            
            print("time: {} s\n".format(time.time() - start))

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val acc: {:4f}, best epoch: {:d}\n'.format(best_val_acc, best_epoch))
        self.model.load_state_dict(best_model_wts)
        self.loss_data = {'best_epoch': best_epoch, 'best_val_acc': best_val_acc, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}
        return self.model, self.loss_data
    

    def test(self, result_num):
        phase = 'test'
        all_x, all_y, all_attn, all_pred, ids = [], [], [], [], set()

        if result_num > len(self.dataloaders[phase].dataset):
            print('The number of results that you want to see are larger than total test set')
            sys.exit()
        
        # statistics of IMDb test set
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            self.model.eval()
            for x, y in self.dataloaders[phase]:
                batch = x.size(0)
                x, y = x.to(self.device), y.to(self.device)

                output, score = self.model(x)
                loss = self.criterion(output, y)
                acc = ((output > 0.5).float()==y).float().sum()/batch

                total_loss += loss.item()*batch
                total_acc += acc.item()*batch

                all_x.append(x.detach().cpu())
                all_y.append(y.detach().cpu())
                all_pred.append(output.detach().cpu())
                if self.config.is_attn:
                    all_attn.append(score.detach().cpu())

            all_x = torch.cat(all_x, dim=0)
            all_y = torch.cat(all_y, dim=0)
            all_pred = torch.cat(all_pred, dim=0)
            if self.config.is_attn:
                all_attn = torch.cat(all_attn, dim=0)
            epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
            epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
            print('{} loss: {:4f}, acc: {:4f}\n'.format(phase, epoch_loss, epoch_acc))

        # visualize the attention score
        if self.config.visualize_attn and self.config.is_attn:
            visualize_attn(all_x, all_y, all_pred, all_attn, self.tokenizer)
  
        # show the sample results
        while len(ids) != result_num:
            ids.add(random.randrange(all_x.size(0)))
        ids = list(ids)
        all_x = torch.cat([all_x[id].unsqueeze(0) for id in ids], dim=0).to(self.device)
        all_y = torch.cat([all_y[id].unsqueeze(0) for id in ids], dim=0).to(self.device)
        output, score = self.model(all_x)

        all_x, all_y, output = all_x.detach().cpu().tolist(), all_y.detach().cpu().tolist(), np.round(output.detach().cpu().tolist(), 3)
        for x, y, pred in zip(all_x, all_y, output):
            print(self.tokenizer.decode(x))
            print('*'*100)
            if pred >= 0.5:
                print('It is positive with a probability of {}'.format(pred))
            else:
                print('It is negative with a probability of {}'.format(1-pred))
            print('ground truth: {}'.format(y))
            print('*'*100)
            print('\n\n')