import random
import numpy as np
import torch as t
import os
import adapkoopnet as model

args = {'path': 'checkponint/true_new_64/', 'f_path': 'fig/vtest/10/'}
# args = {'path': 'checkponint/true_new_nonlinear/', 'f_path': 'fig/vtest/10/'}
if not os.path.exists(args['path']):
    os.makedirs(args['path'])
if not os.path.exists(args['f_path']):
    os.makedirs(args['f_path'])

# ---------------------------------------------------------------------------
# 参数设置
seed = 72
random.seed(30)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
# device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
device = t.device("cpu")
learning_rate = 0.0005
dataset = "highd"  # highd ngsim
args['train_flag'] = True
args['loss2'] = 2
args['loss3'] = 10
args['gamma'] = 0.6  # 学习率衰减
args['num_worker'] = 8
args['device'] = device
args['lstm_encoder_size'] = 64
args['n_head'] = 4
args['att_out'] = 32
args['in_length'] = 31
args['out_length'] = 15
args['f_length'] = 5
args['traj_linear_hidden'] = 64
args['batch_size'] = 256
args['use_elu'] = True
args['dropout'] = 0.2
args['relu'] = 0.1
args['liner_dec'] = 0
args['train_flag'] = False
args['epoch'] = 20

args['use_mse'] = False
args['val_use_mse'] = True

# -------------------------------------------------------------------------
args['np'] = 10

args['veh_num'] = 10
args['time_step'] = 0.12
args['sim_step'] = 1550
args['v_max'] = 33
args['v_min'] = 0
args['a_max'] = 6
args['a_min'] = -6
args['h_max'] = 150
args['h_min'] = 0

args['x_weight'] = 10
args['v_weight'] = 10
args['u_weight'] = 1
args['dv_weight'] = 100

args['a_idm'] = [1.3258, 1.500000017959308, 1.3258, 1.500000017959308, 1.3258]
args['ve'] = [35.9551, 54.246295638178570, 35.9551, 54.246295638178570, 35.9551]
args['b_idm'] = [4, 6, 6, 5.999999994085523, 6]
args['s0'] = [8.1645, 9.657302537298452, 8.1645, 9.657302537298452, 8.1645]
args['T0'] = [1.3318, 1.714745062310631, 1.3318, 1.714745062310631, 1.3318]
args['delta'] = [4, 4, 4, 4, 4]
args['veh_length'] = [4.24, 11.82, 4.24, 11.82, 4.24]
args['cav_th'] = 1
args['cav_permeability'] = 0.2
args['truck_permeability'] = 0.2

# -------------------------------------------------------------------------
