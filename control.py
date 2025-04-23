from __future__ import print_function
import numpy as np
import loader2 as lo
from torch.utils.data import DataLoader
import pandas as pd
from config import *
from KMPC.rs import *
import matplotlib.pyplot as plt
import random
import os
import time
import copy
from scipy.optimize import minimize
import torch as t
import scipy.io as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if not os.path.isfile('test_dx.csv'):
    with open('test_dx.csv', "w") as file:
        file.write("Header1\n")



class simulate():

    def __init__(self, args):
        self.veh_length = None
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1
        self.veh_num = args['veh_num']
        self.device = args['device']
        self.dt = args['time_step']
        self.ss = args['sim_step']
        self.in_length = args['in_length']
        self.v_max = args['v_max']
        self.v_min = args['v_min']
        self.n_p = args['np']
        self.th = args['cav_th']

    def dxMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        dx_pred = y_pred[:, :, 1]
        dx = y_gt[:, :, 1]
        out = t.pow(dx_pred - dx, 2)
        acc[:, :, 0] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)

        return lossVal, counts

    def initial(self, v_ref, plat_arr):
        v_ref = v_ref[:self.ss]
        Y_ref = np.tile(v_ref, (2, 1)).T
        Y_ref[0, 0] = 0
        new_state = re_state(args, plat_arr)
        H = []
        for i in range(len(v_ref) - 1):
            Y_ref[i + 1, 0] = Y_ref[i, 0] + v_ref[i] * self.dt
        for i in range(self.in_length):
            h_eq = new_state.IDM_eq(v_ref[i])
            H.append(h_eq[0])

        Y_0 = np.zeros_like(Y_ref)
        Y_0[0:self.in_length, :] = Y_ref[0:self.in_length, :]
        Y_0[0:self.in_length, 0] = Y_0[0:self.in_length, 0] - H  # CAV初始状态
        Z = np.zeros((self.veh_num, self.ss, 5))
        H_EQ, VEH_L = new_state.Initial_state(v_ref[0])
        Z[:, :, 0] = VEH_L.reshape(-1, 1) * np.ones((self.veh_num, self.ss))
        Z[:, 0, 2] = H_EQ
        Z[:, 0, 3] = v_ref[0]

        return Z, Y_ref, Y_0

    def main(self, name):
        # nedc_v = scp.loadmat('../data/nedc.mat')['vel']
        #
        # v_ref = nedc_v[:self.ss, 0]

        index = np.arange(self.ss)

        # # 前 200 个元素取值为 25
        v_ref = np.ones(self.ss) * 25
        v_ref[40:] = 25 - 5 * np.sin(0.02 * (index[40:] - 40))


        k_m = koopman_matrix(args)
        A, B, H, F_S, F_R, plat_arr, num_cav, num_hdv, num_truck = k_m.main()
        hdv_index = np.where(plat_arr != 0)
        cav_index = np.where(plat_arr == 0)
        car_index = np.where(plat_arr == 1)
        truck_index = np.where(plat_arr == 2)
        k_o = koop_optimize(args, cav_index)
        rs = re_state(args, plat_arr)
        self.veh_length = rs.main()

        l_path = args['path']
        args['train_flag'] = False
        

        dsEncoder = model.DSEncoder(args)
        stEncoder = model.StateEncoder(args)
        Encoder = model.Encoder(args)

        dsEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_ds.tar', map_location=self.device))
        stEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_st.tar', map_location=self.device))

        Encoder.load_state_dict(t.load(l_path + '/epoch' + name + '_e.tar', map_location=self.device))
        dsEncoder = dsEncoder.to(device)
        stEncoder = stEncoder.to(device)
        Encoder = Encoder.to(device)
        dsEncoder.eval()
        stEncoder.eval()
        Encoder.eval()

        Z, Y_ref, Y_0 = self.initial(v_ref, plat_arr)

        x0 = np.random.uniform(-2, 2, self.n_p * num_cav)
        U = np.zeros([v_ref.shape[0], num_cav])
        ct = np.zeros_like(v_ref)

        all_time = 0

        with(t.no_grad()):
            for i in range(self.ss - self.n_p):
                print(i)

                # HDV加速度

                a_t_hdv = rs.IDM_cf_model(Z[hdv_index, i, 1], Z[hdv_index, i, 2], Z[hdv_index, i, 3])
                # a_t_cav = np.ones(num_cav)
                # a_t = np.zeros(self.veh_num)
                # a_t[hdv_index] = a_t_hdv

                Z[hdv_index, i, 4] = a_t_hdv
                # CAV控制输入
                if i >= self.in_length - 1:
                    # ze = t.from_numpy(Z[hdv_index, i - self.in_length + 1:i + 1, :]).to(t.float32).to(device).squeeze(0)
                    ze = t.from_numpy(Z[hdv_index, i - self.in_length + 1:i + 1, :]).to(t.float32).squeeze(0)
                    y_ref = np.zeros([self.veh_num, self.n_p, 3])

                    # y_ref[:, 1] = np.mean(Y_ref[i + 1:i + self.n_p + 1, 1])
                    # for j in range(self.veh_num):
                    #     y_ref[:, 4 + 3 * j] = y_ref[:, 1]

                    y_ref[:, :, 1] = np.mean(v_ref[i - self.n_p:i])
                    # y_ref[cav_index[0][1:], :, 1] = Z[cav_index[0][1:] - 1, i - self.n_p:i, -2]
                    for rei in range(1, len(cav_index[0])):

                        y_ref[cav_index[0][rei]:, :, 1] = np.mean(Z[cav_index[0][rei] - 1, i - self.n_p:i, 3])
                    # for j in range(len(cav_index[0])-1):
                    #     y_ref[cav_index[0][j]+1:cav_index[0][j+1], :, 1] = y_ref[cav_index[0][j], :, 1]
                    # y_ref[cav_index[0][-1]:, :, 1] = y_ref[cav_index[0][-1], :, 1]
                    # y_ref[cav_index[0][1:], :, 1] = np.mean(v_ref[i + 1 - self.n_p:i + 1])
                    y_ref[0, :, 0] = Y_ref[i + 1:i + self.n_p + 1, 0]
                    # y_ref[:, :2] = Y_ref[i:i + self.n_p , :]
                    # y_ref[:, 0] = y_ref[:, 0] - (self.veh_length[0] + self.th * v_ref[i:i + self.n_p])
                    y_ref = y_ref.swapaxes(0, 1).flatten()



                    ds = dsEncoder(ze)
                    st = stEncoder(ze[:, :, 1:-1])
                    # st = stEncoder(ze[:, -1:, 1:-1])
                    # ks = st
                    ks = Encoder(st, ds)
                    # KS = t.squeeze(ks).cpu().numpy()
                    KS = t.squeeze(ks).numpy()
                    kt = {k: v for k, v in zip(hdv_index[0], KS)}
                    kt_cav = {k: v for k, v in zip(cav_index[0], Z[cav_index, i:i + 1, 2:].reshape(-1, 3))}

                    kt.update(kt_cav)
                    # kt = np.empty(self.veh_num, dtype=object)
                    # kt[hdv_index] = KS
                    # kt[cav_index] = Z[hdv_index, i, 2:]

                    s0 = np.concatenate([kt[j] for j in range(self.veh_num)])
                    s0[0] = Y_0[i, 0]

                    # obj_fun = k_o.set_obj(s0, y_ref, A, B, D, k_m.Q, k_m.R)
                    obj_fun = k_o.set_obj(s0, y_ref, H, F_S, F_R)
                    bounds = k_o.set_bounds(num_cav)

                    cons = k_o.set_cons(A, B, s0, y_ref)
                    #
                    start_time = time.time()
                    # results = minimize(obj_fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
                    results = minimize(obj_fun, x0, method='SLSQP', bounds=bounds)
                    ct[i] = time.time() - start_time

                    x = results.x
                    test_state = np.dot(A, s0) + np.dot(B, x)
                    # q = opt_q.set_q(s0, y_ref, A, B, k_m.Q)
                    # P = matrix(H)
                    # q = matrix(q)
                    # results = qp(P, q, None, None, None, None)
                    # x = results['x']
                    # numpy_array = np.array(x)
                    # if abs(x[0]) < 1e-8:
                    #     x[0] = 0.0
                    U[i] = x[:num_cav]

                    # x0 = np.concatenate([x[1:], [random.random() * 0.5]])
                    x0 = x
                    Z[cav_index, i + 1, 4] = Z[cav_index, i, 4] + self.dt * U[i]
                    Y_0[i + 1, 1], Y_0[i + 1, 0] = rs.lead_motion(Z[0, i, 4], U[i, 0], Y_0[i, 1], Y_0[i, 0])

                # HDV状态更新

                V_t_next, delta_V_t_next, H_t_next = rs.veh_state(Z[:, i, 3], Z[:, i, 2], v_ref[i + 1], Z[:, i, -1])
                Z[:, i + 1, 3] = V_t_next
                Z[:, i + 1, 1] = delta_V_t_next
                Z[:, i + 1, 2] = H_t_next

        pos = np.zeros([Z.shape[0], Z.shape[1], 1])
        pos[0, :, 0] = Y_0[:, 0]
        for i in range(1, self.veh_num):
            pos[i, :, 0] = pos[i - 1, :, 0] - Z[i, :, 2]
        Z = np.concatenate([Z, pos - pos[-1, 0, 0]], axis=-1)

        print(np.mean(ct))
        print(np.var(Z[:, 100:, 3]))
        print(np.var(Z[1:, 100:, 2]))




if __name__ == '__main__':
    names = ['8']

    sim = simulate(args)
    # for epoch in names:
    sim.main(name='8')
