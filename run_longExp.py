import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.str2bool import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    
    # AirPhyNet
    parser.add_argument('--adjacency_filename', type=str, default='/remote-home/jiangxudong/FRNet-main/dataset/Air/stations_yrd.csv', help='adjacency_filename')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=96, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    # ModernTCN
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=1, help='ffn_ratio')
    #parser.add_argument('--patch_size', type=int, default=8, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=4, help='the patch stride')
    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1], help='num_blocks in each stage')
    parser.add_argument('--dims', nargs='+',type=int, default=[16,16,16,16], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[16,16,16,16])
    parser.add_argument('--large_size', nargs='+',type=int, default=[35], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5], help='small kernel size for structral reparam')
    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')

    #p.arser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=str2bool, default=False, help='use_multi_scale fusion')  

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

    parser.add_argument('--modes', type=int, default=32, help='modes to be selected random 64')
    parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
    # parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    
    parser.add_argument('--down_sampling_method', type=str, default='avg',help='down sampling method, only support avg, max, conv')
    parser.add_argument('--channel_independence', type=int, default=1,help='0: channel dependence 1: channel independence for FreTS model')    
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')    
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    #FRNet
    parser.add_argument('--pred_head_type', type=str, default='linear', help='linear or truncation')
    parser.add_argument('--aggregation_type', type=str, default='linear', help='linear or avg')
    parser.add_argument('--channel_attention', type=int, default=0, help='True 1 or False 0')
    parser.add_argument('--global_freq_pred', type=int, default=1, help='True 1 or False 0')
    parser.add_argument('--period_list', type=int, nargs='+', default=1, help='period_list') 
    parser.add_argument('--emb', type=int, default=64, help='patch embedding size')

    #AdaMamba
    parser.add_argument('--beta', type=float, default=1, help='beta')
    parser.add_argument('--ckernel', type=int, default=3, help='ckernel')
    parser.add_argument('--dim_feq', type=int, default=256, help='dimension of model')
    parser.add_argument('--dim_pitch', type=int, default=256, help='dimension of model')
    parser.add_argument('--patch_lens', type=str, default='96,72,48,36,18,9', help='comma separated list of patch lengths for multi-scale')
    
    #AdaMamba
    parser.add_argument('--grid_size', type=int, default=5, help='grid_size in KAN')  # KAN hyperparam: grid for splines (tune for PDE accuracy)
    parser.add_argument('--spline_order', type=int, default=3, help='spline_order in KAN') # Higher for smoother functions

    #AdaMamba
    parser.add_argument('--dt_rank', type=int, default=32)
    parser.add_argument('--dt_min', type=float, default=0.001)
    parser.add_argument('--dt_init', type=str, default='random', help='random or constant')
    parser.add_argument('--dt_max', type=float, default=0.1)
    parser.add_argument('--dt_scale', type=float, default=1.0)
    parser.add_argument('--dt_init_floor', type=float, default=1e-4)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--conv_bias', type=bool, default=True)
    parser.add_argument('--pscan', action='store_true', help='use parallel scan mode or sequential mode when training', default=True)
    parser.add_argument('--d_state', type=int, default=16, help='parameter of Mamba Block')
    # parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    # parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
    parser.add_argument('--n1',type=int,default=512,help='First Embedded representation')
    parser.add_argument('--n2',type=int,default=256,help='Second Embedded representation')
    parser.add_argument('--ch_ind', type=int, default=0, help='Channel Independence; True 1 False 0')
    parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
    parser.add_argument('--LDkernel_size', type=int, default=25, help='kernel_size hyperparameter of smoothing')
    parser.add_argument('--n_layers', type=int, default=3, help='n_layers of DEFT Block')

    # Affirm
    parser.add_argument('--d_conv_1', type=int, default=2, help='d_conv parameter of Mamba')
    parser.add_argument('--d_conv_2', type=int, default=4, help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
       
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    parser.add_argument('--checkpoint_save', action='store_true', default=False, help='delete checkpoint')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_patchlen{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.patch_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        