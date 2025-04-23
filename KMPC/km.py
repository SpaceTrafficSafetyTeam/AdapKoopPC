

from config import *


class koopman_matrix:

    def __init__(self, args):
        self.path = args['path']
        self.device = args['device']
        self.n_p = args['np']
        self.veh_num = args['veh_num']
        self.koopstate = args['lstm_encoder_size']
        self.dt = args['time_step']
        self.cav_perme = args['cav_permeability']
        self.truck_perme = args['truck_permeability']
        self.plat_arr, self.num_cav, self.num_hdv, self.num_truck = self.generate_arrangement()
        self.a_hdv = np.zeros((self.koopstate, self.koopstate))
        self.b_hdv = np.zeros((self.koopstate, 1))
        self.c_hdv = np.zeros((3, self.koopstate))
        self.a_cav = np.eye(3)
        self.a_cav[0, 1] = -self.dt
        self.a_cav[1, 2] = self.dt

        self.b_cav = np.array([[0], [0.5 * self.dt ** 2], [self.dt]])

        self.c_cav = np.eye(3)
        aaa = (1.0 - self.cav_perme)
        self.k_dim = int(self.koopstate * self.veh_num * round(1 - self.cav_perme, 2) + 3 * self.veh_num * self.cav_perme)

        self.d = (3 * self.veh_num) * self.n_p
        self.A = np.zeros((self.k_dim, self.k_dim))
        self.A[0:3, 0:3] = self.a_cav
        self.A[0, 1] = self.dt
        self.A[0, 2] = 0.5 * self.dt ** 2
        self.B = np.zeros((self.k_dim, self.num_cav))
        self.B[1, 0] = 0.5 * self.dt ** 2
        self.B[2, 0] = self.dt
        self.C = np.zeros((3 * self.veh_num, self.k_dim))

        self.C[0:3, 0:3] = self.c_cav

        # weight
        self.x_weight = args['x_weight']
        self.v_weight = args['v_weight']
        self.dv_weight = args['dv_weight']
        self.u_weight = args['u_weight']
        self.R = np.eye(self.n_p * self.num_cav) * self.u_weight
        self.q_cav = np.diag([0, self.v_weight, 0])
        self.q_hdv = np.diag([0, 0, self.dv_weight])
        self.Q = np.zeros((self.d, self.d))

        for i in np.where(self.plat_arr == 0)[0]:
            # for j in [i - 1, i + 1]:
            #     if j not in np.where(self.plat_arr == 0)[0] and 0 < j <= self.veh_num-1:
            #         self.Q[j * 3:(j + 1) * 3, j * 3:(j + 1) * 3] = self.q_hdv
            self.Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self.q_cav

        # self.Q[0, 0] = self.x_weight
        for i in np.where(self.plat_arr != 0)[0]:
            self.Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self.q_hdv
        for j in range(1, self.n_p):
            self.Q[3 * self.veh_num * j:3 * self.veh_num * (j + 1), 3 * self.veh_num * j:3 * self.veh_num * (j + 1)] \
                = self.Q[0:3 * self.veh_num, 0:3 * self.veh_num]

    def generate_arrangement(self):
        num_cav = int(self.veh_num * self.cav_perme)
        num_truck = int(self.veh_num * self.truck_perme)
        num_hdv = self.veh_num - num_cav - num_truck

        # cav位置
        # arr1 = [0] * 2 + [1] * 11 + [2] * 3
        # arr2 = [0] * 2 + [1] * 12 + [2] * 4
        # arr3 = [0] * 1 + [1] * 12 + [2] * 3
        # arr1_rand = arr1[1:]
        # random.shuffle(arr1_rand)
        # random.shuffle(arr2)
        # random.shuffle(arr3)
        # arr = [arr1[0]] + arr1_rand + arr2 + arr3

        # 生成只包含0和1的列表，0的数量由 zero_ratio 决定
        arr = [0] * num_cav + [1] * num_hdv + [2] * num_truck
        arr_rand = arr[1:]
        # 随机打乱列表，以确保随机性
        random.shuffle(arr_rand)
        arr = [arr[0]]+arr_rand
        # arr =[0, 1, 1, 1, 1, 1, 0, 1, 1, 1] * 5
        # 转换为 NumPy 数组
        # arr[23] = 2
        plat_arr = np.array(arr)

        return plat_arr, num_cav, num_hdv, num_truck

    def abc(self):
        name = '8'

        decoder = model.Koop_space(args)

        decoder.load_state_dict(t.load(self.path + '/epoch' + name + '_k.tar', map_location=self.device))
        weights = decoder.state_dict()

        self.a_hdv = weights['A.weight'].numpy()
        self.b_hdv = weights['B.weight'].numpy()
        self.c_hdv = weights['C.weight'].numpy()[[1, 2, 0]]

    def generate_index_array(self):
        index_array = []
        current_index = 0
        index_array.append(current_index)
        for element in self.plat_arr:

            if element == 0:
                current_index += 3
            else:
                current_index += self.koopstate
            index_array.append(current_index)
        return np.array(index_array)

    def ABC(self):
        self.index = self.generate_index_array()
        j = 1

        for i in range(1, self.veh_num):
            if self.plat_arr[i] == 0:
                self.A[self.index[i]:self.index[i + 1], self.index[i]:self.index[i + 1]] = self.a_cav
                if self.plat_arr[i - 1] == 0:
                    self.A[self.index[i], self.index[i] - 2] = self.dt
                else:
                    self.A[self.index[i], self.index[i - 1]:self.index[i]] = self.dt * self.c_hdv[1:-1]
                # if i != 0:
                #     self.A[self.index[i], self.index[i] - 2] = self.dt

                self.B[self.index[i]:self.index[i + 1], j:j + 1] = self.b_cav
                self.C[3 * i:3 * (i + 1), self.index[i]:self.index[i + 1]] = self.c_cav
                j += 1
            else:
                self.A[self.index[i]:self.index[i + 1], self.index[i]:self.index[i + 1]] = self.a_hdv
                self.C[3 * i:3 * (i + 1), self.index[i]:self.index[i + 1]] = self.c_hdv
                if i >= 1:
                    if self.plat_arr[i - 1] == 0:
                        self.A[self.index[i]:self.index[i + 1], self.index[i] - 2:self.index[i] - 1] = self.b_hdv
                    else:
                        self.A[self.index[i]:self.index[i + 1], self.index[i - 1]:self.index[i]] = np.dot(self.b_hdv,
                                                                                                          self.c_hdv[
                                                                                                          1:-1])

    def S_A(self):
        S_A = np.empty((0, self.A.shape[1]))
        for i in range(self.n_p):
            # t = np.linalg.matrix_power(self.A, (i + 1))
            temp = np.dot(self.C, np.linalg.matrix_power(self.A, (i + 1)))
            S_A = np.concatenate((S_A, temp), axis=0)

        return S_A

    def S_B(self):
        S_B = np.zeros((self.d, self.n_p * self.num_cav))
        for i in range(self.n_p):
            for j in range(i + 1):
                S_B[i * self.C.shape[0]:(i + 1) * self.C.shape[0], j * self.num_cav:(j + 1) * self.num_cav] = np.dot(
                    np.dot(self.C, np.linalg.matrix_power(self.A, i - j)), self.B)

        return S_B

    # def S_D(self):
    #     s_d = np.dot(self.C, self.D_AB) + self.D_C
    #     S_D = np.tile(s_d, (self.n_p, 1))
    #
    #     return S_D

    def weight_matrix(self, A, B):
        H = 2 * (np.dot(np.dot(B.T, self.Q), B) + self.R)
        F_S = 2 * np.dot(np.dot(A.T, self.Q), B)
        F_R = 2 * np.dot(self.Q, B)
        return H, F_S, F_R

    def main(self):
        self.abc()
        self.ABC()
        # eigenvalues = np.linalg.eigvals(self.A)
        # Co = np.hstack((self.B, np.dot(self.A, self.B), np.dot(np.dot(self.A, self.A), self.B)))
        #
        # # 计算可控性矩阵的秩
        # controllability_rank = np.linalg.matrix_rank(Co)
        # print(controllability_rank)
        A = self.S_A()
        B = self.S_B()
        # D = self.S_D()
        H, F_S, F_R = self.weight_matrix(A, B)
        return A, B, H, F_S, F_R, self.plat_arr, self.num_cav, self.num_hdv, self.num_truck
