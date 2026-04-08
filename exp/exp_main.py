from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FRNet
from models import AdaMamba
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
# 调用工具类中的可视化函数
from utils.tools import visualize_frequency_adaptation
from utils.metrics import metric
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from matplotlib.colors import LinearSegmentedColormap

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
    # def _build_model(self):
    #     model_dict = {

    #         'FRNet':FRNet
    #     }

    def _build_model(self):
        model_dict = {
            'AdaMamba':AdaMamba
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total params(M): {total_params/1e6}")
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _select_mae_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'AdaMamba' in self.args.model:
                            outputs, _ = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'AdaMamba' in self.args.model:
                        outputs, _ = self.model(batch_x)                    
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_mae = self._select_mae_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'AdaMamba' in self.args.model:
                            outputs, outputs_phy = self.model(batch_x)                        
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if 'AdaMamba' in self.args.model:
                            physics_loss = self.model.module.loss_function(outputs,outputs_phy)
                            loss = loss + physics_loss
                            # physics_loss = torch.mean((outputs - outputs_phy) ** 2)
                            # loss = loss + 0.1 * physics_loss
                        train_loss.append(loss.item())
                else:
                    if 'AdaMamba' in self.args.model:
                            outputs, outputs_phy = self.model(batch_x)                    
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)              
                    if 'AdaMamba' in self.args.model:
                        #physics_loss = self.model.module.loss_function(outputs,outputs_phy)
                        loss = criterion(outputs, batch_y) #+ physics_loss
                        # physics_loss = torch.mean((outputs - outputs_phy) ** 2)
                        # loss = criterion(outputs, batch_y) + 0.1 * physics_loss
                    else:
                        loss = criterion(outputs, batch_y) 
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}h'.format(speed, left_time/60/60))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}min".format(epoch + 1, (time.time() - epoch_time)/60))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_mae_loss = self.vali(test_data, test_loader, criterion_mae)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test MSE Loss: {4:.7f} Test MAE: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, test_mae_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def show_heatmap(self, data):

        sns.set_style('whitegrid')  
        ax=plt.figure(figsize=(30, 30),dpi=100)
        sns.heatmap(data, linewidths=0, square=True,linecolor='white',annot=False, cbar =False, cmap = 'Oranges')
        plt.show()
        plt.savefig('./attention.pdf')

    def visualize_frequency_activation(self, A_tensor, omega_tensor, setting, batch_idx, dim_feq, folder_path):
        """可视化单个序列的频率幅度 A 热力图。"""

        attn_pink = LinearSegmentedColormap.from_list(
            'attn_pink',
            [
                (0.0, '#ffffff'),  
                (0.3, '#fde0ef'),  
                (0.6, '#f768a1'), 
                (1.0, '#7a0177')   
            ]
        )
        
        A = A_tensor.mean(dim=1).cpu().numpy() # [nsteps, dim_feq]
        omega = omega_tensor.cpu().numpy()
        nsteps = A.shape[0]
        
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        im = plt.imshow(A.T, aspect='auto', origin='lower', cmap=attn_pink, 
                extent=[0, nsteps, 0, dim_feq], interpolation='none')
        
        cbar = plt.colorbar(im)
        cbar.set_label(
            'Frequency Amplitude A (Mean over Features)',
            fontsize=25
        )
        cbar.ax.tick_params(labelsize=27)

        ax = plt.gca()
        step = max(1, dim_feq // 10) 
        y_ticks = np.arange(0, dim_feq, step)
       
        ax.set_yticks(y_ticks + 0.5) # +0.5 放置在格子的中间
        y_labels = [f'{omega[int(i)]:.2f}' for i in y_ticks]
        ax.set_yticklabels(y_labels)
        plt.tick_params(axis='both', labelsize=35)
        plt.savefig(os.path.join(folder_path, f'{batch_idx}_A_heatmap.pdf'))
        plt.close()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                A_batch = None
                omega_batch = None
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'AdaMamba' in self.args.model:
                            outputs, outputs_phy = self.model(batch_x)   
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'AdaMamba' in self.args.model:

                        return_A = (i % 10 == 0 and i <= 40)

                        if return_A:
                            outputs, outputs_phy, A_batch, omega_batch, omega_offsets_batch, g_batch = self.model(batch_x, return_A=True) 
                            print(f"[{i}] Extracting frequency analysis features for visualization...")

                            A_sample = A_batch[0].detach().cpu().numpy() # [nsteps, dim, dim_feq]
                            omega_offsets_sample = omega_offsets_batch[0].detach().cpu().numpy() # [dim_feq]
                            g_sample = g_batch[0].detach().cpu().numpy() # [nsteps, dim, dim_feq]
                            
                            save_file = os.path.join(folder_path, f'{i}_frequency_analysis.pdf')
                            visualize_frequency_adaptation(omega_offsets_sample, g_sample, A_sample, save_file)
                            print(f"✅ Frequency analysis heatmap saved for Batch {i} at {save_file}")
                        else:
                            outputs, outputs_phy = self.model(batch_x, return_A=False)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)



                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                  
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # gt = true[0, :, -1] #np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = pred[0, :, -1] #np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # print(f"gt shape: {gt.shape}")
                    # print(f"pd shape: {pd.shape}")
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, mape:{}, corr:{}'.format(mse, mae, mape, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{}, corr:{}'.format(mse, mae, mape,corr))
        f.write('\n')
        f.write('\n')
        f.close()


        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'FRNet' in self.args.model:
                            # outputs, trend, seasonal, decomp_module = self.model(batch_x)
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'FRNet' in self.args.model:
                        # outputs, trend, seasonal, decomp_module = self.model(batch_x)
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
