"""
@Coding: uft-8
@Time: 2019-05-27 14:32
@Author: Ryne Chen
@File: support.py 
"""

import numpy as np
from Ch06_SVM import svm_SMO


class optStruct:
    def __init__(self, data_in, labels, C, tolerance):
        self.X = data_in
        self.label_matrix = labels
        self.C = C
        self.tolerance = tolerance
        self.m = np.shape(data_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))


def cal_Ek(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.label_matrix).T * oS.X * oS.X[k, :].T + oS.b)
    Ek = fXk - float(oS.label_matrix[k])
    return Ek


def select_J(i, oS, Ei):
    max_K = -1
    max_delta_E = 0
    Ej = 0

    oS.e_cache[i] = [1, Ei]
    valid_E_cache_list = np.nonzero(oS.e_cache[:, 0].A)[0]

    if len(valid_E_cache_list) > 1:
        for k in valid_E_cache_list:
            if k == i:
                continue
            Ek = cal_Ek(oS, k)
            delta_E = abs(Ei - Ek)
            if delta_E > max_delta_E:
                max_K = k
                max_delta_E = delta_E
                Ej = Ek
        return max_K, Ej

    else:
        j = svm_SMO.select_Jrand(i, oS.m)
        Ej = cal_Ek(oS, j)
    return j, Ej


def update_Ek(oS, k):
    Ek = cal_Ek(oS, k)
    oS.e_cache[k] = [1, Ek]


def inner_L(i, oS):
    Ei = cal_Ek(oS, i)

    if (oS.label_matrix[i] * Ei < -oS.tolerance and oS.alphas[i] < oS.C) or \
            (oS.label_matrix[i] * Ei > -oS.tolerance and oS.alphas[i] > 0):
        j, Ej = select_J(i, oS, Ei)
        alpha_I_old = oS.alphas[i].copy()
        alpha_J_old = oS.alphas[j].copy()

        if oS.label_matrix[i] != oS.label_matrix[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])

        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print('L == H')
            return 0

        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T

        if eta >= 0:
            print('eta >= 0')
            return 0

        oS.alphas[j] -= oS.label_matrix[j] * (Ei - Ej) / eta
        oS.alphas[j] = svm_SMO.clip_alpha(oS.alphas[j], H, L)

        update_Ek(oS, j)

        if abs(oS.alphas[j] - alpha_J_old) < 0.00001:
            print('j not moving enough')
            return 0

        oS.alphas[i] += oS.label_matrix[j] * oS.label_matrix[i] * (alpha_J_old - oS.alphas[j])

        update_Ek(oS, i)

        b1 = oS.b - Ei - oS.label_matrix[i] * (oS.alphas[i] - alpha_I_old) * oS.X[i, :] * oS.X[i, :].T - \
             oS.label_matrix[j] * (oS.alphas[j] - alpha_J_old) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.label_matrix[i] * (oS.alphas[i] - alpha_I_old) * oS.X[i, :] * oS.X[j, :].T - \
             oS.label_matrix[j] * (oS.alphas[j] - alpha_J_old) * oS.X[j, :] * oS.X[j, :].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
