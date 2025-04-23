import numpy as np

from KMPC.ko import *


class re_state:
    def __init__(self, args, plat_arr):
        self.veh_num = args['veh_num']
        self.dt = args['time_step']
        self.cav_perme = args['cav_permeability']
        self.plat_arr = plat_arr
        self.a_idm = np.zeros([int(self.veh_num * (1 - self.cav_perme))])
        self.ve = np.zeros_like(self.a_idm)
        self.b_idm = np.zeros_like(self.a_idm)
        self.s0 = np.zeros_like(self.a_idm)
        self.T0 = np.zeros_like(self.a_idm)
        self.delta = args['delta'][0]
        self.veh_length = np.zeros_like(self.a_idm)
        hdv_index = self.plat_arr[np.where(self.plat_arr != 0)]
        for i in range(2):
            self.a_idm[hdv_index == i + 1] = args['a_idm'][i]
            self.ve[hdv_index == i + 1] = args['ve'][i]
            self.b_idm[hdv_index == i + 1] = args['b_idm'][i]
            self.s0[hdv_index == i + 1] = args['s0'][i]
            self.T0[hdv_index == i + 1] = args['T0'][i]
            self.veh_length[hdv_index == i + 1] = args['veh_length'][i]

    def main(self):
        return self.veh_length

    def IDM_cf_model(self, delta_V_t, H_t, V_t):
        def desired_space_hw(V_n_t, delta_V_n_t):
            item1 = self.s0
            item2 = np.multiply(self.T0, V_n_t)
            item3 = np.divide(np.multiply(V_n_t, delta_V_n_t), 2 * np.sqrt(np.multiply(self.a_idm, self.b_idm)))

            return item1 + np.maximum(0, item2 - item3)

        desired_S = desired_space_hw(V_t, delta_V_t)
        a_t = np.multiply(self.a_idm, (
                1 - np.power(np.divide(V_t, self.ve), self.delta) - np.divide(desired_S, H_t - self.veh_length) ** 2))

        return a_t

    def IDM_eq(self, V_t):
        def desired_space_hw(V_n_t):
            item1 = self.s0
            item2 = np.multiply(self.T0, V_n_t)

            return item1 + np.maximum(0, item2)

        desired_S = desired_space_hw(V_t)
        h_eq = np.divide(desired_S, np.sqrt(1 - np.power(np.divide(V_t, self.ve), self.delta))) + self.veh_length

        return h_eq

    # def HDV_motion(self, a_t, V_t, H_t, v_cav_t_next):
    #     V_t_next = V_t + a_t * self.dt
    #     delta_V_t_next = np.concatenate(([v_cav_t_next], V_t_next[:-1])) - V_t_next
    #     H_t_next = H_t + delta_V_t_next * self.dt
    #
    #     return V_t_next, delta_V_t_next, H_t_next

    def CAV_motion(self, a_t, u_t):
        a_t_next = a_t + u_t * self.dt
        # V_t_next = V_t + a_t * self.dt + u_t * 0.5 * self.dt ** 2
        #
        # return a_t_next, V_t_next

        return a_t_next

    def veh_state(self, V_t, H_t, v0, a_t):
        V_t_next = V_t + a_t * self.dt
        delta_V_t_next = np.concatenate(([v0], V_t_next[:-1])) - V_t_next
        H_t_next = H_t + delta_V_t_next * self.dt

        return V_t_next, delta_V_t_next, H_t_next

    def Initial_state(self, V_t):
        h_eq = self.IDM_eq(V_t)
        H_EQ = np.zeros(self.veh_num)
        H_EQ[self.plat_arr == 1] = np.min(h_eq)
        H_EQ[self.plat_arr == 0] = np.min(h_eq)
        H_EQ[self.plat_arr == 2] = np.max(h_eq)
        VEH_L = np.zeros(self.veh_num)
        VEH_L[self.plat_arr != 2] = np.min(self.veh_length)
        VEH_L[self.plat_arr == 2] = np.max(self.veh_length)

        return H_EQ, VEH_L

    def lead_motion(self, a_t, u_t, V_t, x_t):
        # a_t_next = a_t + u_t * self.dt
        V_t_next = V_t + a_t * self.dt + u_t * 0.5 * self.dt ** 2
        x_t_next = x_t + V_t * self.dt
        return V_t_next, x_t_next


