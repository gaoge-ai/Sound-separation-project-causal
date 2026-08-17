
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from thop import profile
from thop import clever_format
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import os
import gc
import datetime
import time
import random
import glob
from tqdm import tqdm
import pandas as pd
import itertools
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("..")
from utils.utils_library_gpu import *
from DNN_models.Complex_MTASS import *





# -----------------------------------------------------------------------------------------------------------------------------
# * Class:
#     Complex_MTASS_model---Implements a Complex-domain MTASS model for speech, noise and music separation 
# 
# * Note:
#   The Complex MTASS model takes the mixture Mag feratures as the inputs 
#   and outputs the complex ratio masks (cRMs).
#   In this model, 8 sub-bands are divided and performed the multi-scale analysis.
#   
#
# * Copyright and Authors:
#    Writen by Mr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
# -----------------------------------------------------------------------------------------------------------------------------

class Complex_MTASS_model:

    #-------------------------------------------------------------------------------------------------------------------
    # * Functions:
    #     model_description()-- print the description of model and the information of training data 
    #        * Arguments:
    #            * train_datain_path1 -- train data path of input
    #            * train_datain_list1 -- train data list of input
    #            * dev_datain_path1 -- dev data path of input
    #            * dev_datain_list1 -- dev data list of output
    #            * mini_batch_size -- the size of each mini_batch
    #        * Returns:
    #            * m_x1_train -- input feature size, shape [0]
    #            * n_x1_train -- input feature size, shape [1]
    #            * num_minibatches_train -- the toal numbers of training data
    #            * num_minibatches_dev -- the total numbers of dev data
    #
    #---------------------------------------------------------------------------------------------------------------------

    def model_description(train_datain_path1,train_datain_list1,dev_datain_path1,dev_datain_list1,mini_batch_size):
        ### START CODE HERE ###
        print('The Complex MTASS learning structure (Mag_to_Com, Residual Compensation, F-MSE+T-SNR) is : 257+ComplexMSTCN(15)+3*GTCN(5,8)+(514,514,514)')
        print('The Complex MTASS model is trained to separate three targets!') 
        print('The sizes of each train/dev file are as follows:')
        num_minibatches_train = 0
        num_minibatches_dev = 0        
        for train_datain in train_datain_list1:
            path = train_datain_path1 + os.sep + train_datain
            data = np.load(path)
            # print('n_x_train', n_x_train)
            num_minibatches_train += math.floor(data.shape[0] / mini_batch_size)
            split_sentence_len = data.shape[2] 

        for dev_datain in dev_datain_list1:
            path = dev_datain_path1 + os.sep + dev_datain
            data = np.load(path)
            # print('n_x_dev', n_x_dev)
            num_minibatches_dev += math.floor(data.shape[0] / mini_batch_size)
                
        print('The mini_batch size is:', mini_batch_size)
        print('num_minibatches_train:', num_minibatches_train)
        print('num_minibatches_dev:', num_minibatches_dev)
    
        del data
        gc.collect()
        return num_minibatches_train, num_minibatches_dev, split_sentence_len 

    def masked_sisdr_loss(estimate, target, eps=1e-8):
        target_energy = torch.sum(target ** 2, dim=-1)
        mask = target_energy > eps
        batch_size = estimate.shape[0]
        loss_vector = torch.zeros(batch_size, device=estimate.device)
        if mask.sum() > 0:
            valid_est = estimate[mask]
            valid_tgt = target[mask]
            valid_sisdr = Complex_MTASS_model.sisdr_cost(valid_est, valid_tgt)
            loss_vector[mask] = -valid_sisdr
        mask_float = mask.float()  
        return loss_vector, mask_float
  
    def compute_out_cost(
        Z1,
        Z2,
        Z3,
        Y_targets,
        R_targets,
        mse_loss_weight=1.0,
        sisdr_loss_weight=1.0,
        l1_loss_weight=0.0,
        magnitude_l1_loss_weight=0.0,
    ):
        ### START CODE HERE ###
        Y1, Y2, Y3 = Y_targets[0], Y_targets[1], Y_targets[2]
        cost_l1 = torch.zeros((), device=Z1.device, dtype=Z1.dtype)
        cost_magnitude_l1 = torch.zeros((), device=Z1.device, dtype=Z1.dtype)
        cost_freq = torch.zeros((), device=Z1.device, dtype=Z1.dtype)
        cost_time_sisdr = torch.zeros((), device=Z1.device, dtype=Z1.dtype)

        if mse_loss_weight != 0:
            mse_cost = torch.nn.MSELoss()
            cost_freq = mse_cost(Z1, Y1) + mse_cost(Z2, Y2) + mse_cost(Z3, Y3)

        if l1_loss_weight != 0:
            l1_cost = torch.nn.L1Loss()
            cost_l1 = l1_cost(Z1, Y1) + l1_cost(Z2, Y2) + l1_cost(Z3, Y3)

        if magnitude_l1_loss_weight != 0:
            l1_cost = torch.nn.L1Loss()
            cost_magnitude_l1 = (
                l1_cost(torch.norm(Z1, dim=1), torch.norm(Y1, dim=1))
                + l1_cost(torch.norm(Z2, dim=1), torch.norm(Y2, dim=1))
                + l1_cost(torch.norm(Z3, dim=1), torch.norm(Y3, dim=1))
            )

        if sisdr_loss_weight != 0:
            win_len = 512
            win_inc = 256 # frame shift
            fft_len = 512
            Z1_time = Complex_MTASS_model.Inverse_STFT(Z1, win_len, win_inc, fft_len)
            Z2_time = Complex_MTASS_model.Inverse_STFT(Z2, win_len, win_inc, fft_len)
            Z3_time = Complex_MTASS_model.Inverse_STFT(Z3, win_len, win_inc, fft_len)
            R1, R2, R3 = R_targets[0], R_targets[1], R_targets[2]

            loss_s, mask_s = Complex_MTASS_model.masked_sisdr_loss(Z1_time, R1)
            loss_m, mask_m = Complex_MTASS_model.masked_sisdr_loss(Z2_time, R2)
            loss_o, mask_o = Complex_MTASS_model.masked_sisdr_loss(Z3_time, R3)
            sum_loss = loss_s + loss_m + loss_o
            num_tasks = mask_s + mask_m + mask_o
            num_tasks = torch.clamp(num_tasks, min=1.0)
            per_sample_loss = sum_loss / num_tasks
            cost_time_sisdr = torch.mean(per_sample_loss)

        total_cost = (
            mse_loss_weight * cost_freq
            + sisdr_loss_weight * cost_time_sisdr
            + l1_loss_weight * cost_l1
            + magnitude_l1_loss_weight * cost_magnitude_l1
        )

        return total_cost, cost_freq, cost_time_sisdr, cost_l1, cost_magnitude_l1

    def sisdr_cost(estimated, target, eps=1e-8):
        dot = torch.sum(estimated * target, dim=-1, keepdim=True)
        s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

        scale = dot / s_energy
        target_scaled = scale * target
        e_noise = estimated - target_scaled
        target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
        noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps
        
        sisdr = 10 * torch.log10(target_pow / noise_pow)
        return sisdr.squeeze(-1)

    def SNR_cost(Z1,Y1,eps=1e-8):
        # Z1.shape=[-1,sen_len]
        # Y1.shape=[-1,sen_len]

        snr = torch.sum(Y1**2, dim=1, keepdim=True) / (torch.sum((Z1 - Y1)**2, dim=1, keepdim=True)+eps)
        loss = -10*torch.log10(snr + eps).mean()
        
        return loss


    def Inverse_STFT(inputs, win_len, win_hop, fft_len):
        # inputs.shape = [B,fea_size,sen_len] (Complex STFT)
        cutoff = fft_len // 2 + 1
        real_part = inputs[:, :cutoff, :]
        imag_part = inputs[:, cutoff:, :]

        complex_spec = torch.complex(real_part, imag_part)
        #istft_window = torch.ones(win_len, device=inputs.device)
        istft_window = torch.hamming_window(win_len, device=inputs.device)

        reconstruction = torch.istft(
            complex_spec,
            n_fft=fft_len,
            hop_length=win_hop,
            win_length=win_len,
            window=istft_window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=False 
        )
        
        return reconstruction
