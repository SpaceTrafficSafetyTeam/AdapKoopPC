from KMPC.km import *


class koop_optimize:
    def __init__(self, args, cav_index):
        self.n_p = args['np']
        self.veh_num = args['veh_num']
        self.d = 3 * self.veh_num * self.n_p
        self.v_max = args['v_max']
        self.v_min = args['v_min']
        self.h_max = args['h_max']
        self.h_min = args['h_min']
        self.a_max = args['a_max']
        self.a_min = args['a_min']
        self.v_fitter = np.zeros((self.d, self.d))
        self.v_fitter[1::3, 1::3] = 1
        self.h_fitter = np.zeros((self.d, self.d))
        self.h_fitter[3:3 * self.veh_num:3, 3:3 * self.veh_num:3] = 1
        for i in range(1, self.n_p):
            self.h_fitter[3 * self.veh_num * i:3 * self.veh_num * (i + 1),
            3 * self.veh_num * i:3 * self.veh_num * (i + 1)] = self.h_fitter[0:3 * self.veh_num, 0:3 * self.veh_num]
        self.h_lead_fitter = np.zeros((self.d, self.d))
        self.h_lead_fitter[0::3 * self.veh_num, 0::3 * self.veh_num] = 1
        self.a_cav_fitter = np.zeros((self.d, self.d))
        self.a_cav_fitter[2 + cav_index[0] * 3, 2 + cav_index[0] * 3] = 1
        for i in range(1, self.n_p):
            self.a_cav_fitter[3 * self.veh_num * i:3 * self.veh_num * (i + 1),
            3 * self.veh_num * i:3 * self.veh_num * (i + 1)] = self.a_cav_fitter[0:3 * self.veh_num, 0:3 * self.veh_num]

    def set_obj(self, y0, y_ref, H, F_S, F_R):
        y0 = y0[:, np.newaxis]
        obj = lambda x: 0.5 * np.dot(np.dot(x.T, H), x) + np.dot(np.dot(y0.T, F_S) - np.dot(y_ref.T, F_R), x)
        return obj

    def set_cons(self, A, B, y0, y_ref):

        # cons = [
        #     {'type': 'ineq', 'fun': lambda x: np.dot(self.v_fitter, np.dot(A, y0) + np.dot(B, x) - self.v_min)},
        #     {'type': 'ineq', 'fun': lambda x: np.dot(self.v_fitter, - (np.dot(A, y0) + np.dot(B, x) - self.v_max))},
        #     {'type': 'ineq', 'fun': lambda x: np.dot(self.h_fitter, np.dot(A, y0) + np.dot(B, x) - self.h_min)},
        #     {'type': 'ineq', 'fun': lambda x: np.dot(self.h_fitter, - (np.dot(A, y0) + np.dot(B, x) - self.h_max))},
        #     {'type': 'ineq',
        #      'fun': lambda x: np.dot(self.h_lead_fitter, y_ref - np.dot(A, y0) - np.dot(B, x) - self.h_min)},
        #     {'type': 'ineq',
        #      'fun': lambda x: np.dot(self.h_lead_fitter, - y_ref + np.dot(A, y0) + np.dot(B, x) + self.h_min)},
        #     {'type': 'ineq', 'fun': lambda x: np.dot(self.a_cav_fitter, np.dot(A, y0) + np.dot(B, x) - self.a_min)},
        #     {'type': 'ineq', 'fun': lambda x: np.dot(self.a_cav_fitter, - (np.dot(A, y0) + np.dot(B, x) - self.a_max))}
        # ]

        cons = [{'type': 'ineq', 'fun': lambda x: np.dot(self.v_fitter, np.dot(A, y0) + np.dot(B, x) - self.v_min)},
                {'type': 'ineq', 'fun': lambda x: np.dot(self.v_fitter, - (np.dot(A, y0) + np.dot(B, x) - self.v_max))},
                {'type': 'ineq', 'fun': lambda x: np.dot(self.h_fitter, np.dot(A, y0) + np.dot(B, x) - self.h_min)},
                {'type': 'ineq', 'fun': lambda x: np.dot(self.h_fitter, - (np.dot(A, y0) + np.dot(B, x) - self.h_max))},
                # {'type': 'ineq',
                #       'fun': lambda x: np.dot(self.h_lead_fitter, y_ref - np.dot(A, y0) - np.dot(B, x) - self.h_min)},
            {'type': 'ineq', 'fun': lambda x: np.dot(self.a_cav_fitter, np.dot(A, y0) + np.dot(B, x) - self.a_min)},
            {'type': 'ineq', 'fun': lambda x: np.dot(self.a_cav_fitter, - (np.dot(A, y0) + np.dot(B, x) - self.a_max))}
                ]
        return cons

    def set_bounds(self, num_cav):
        bounds = [(self.a_min, self.a_max)] * self.n_p * num_cav
        return bounds
