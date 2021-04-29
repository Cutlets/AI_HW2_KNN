# 홍익대학교 컴퓨터공학과 2021년 1학기 인공지능 HW2
# 학  번 : B511156
# 이  름 : 이 치 현

import numpy as np

# knn.py
###################################################


class SimKnn:
    kdata = []
    klabel = []
    knei = 0

    def __init__(self, idata, ilabel, inei):
        self.kdata = idata
        self.klabel = ilabel
        self.knei = inei

    def e_distance(self, dsource, dinput):
        return float(np.sqrt(np.sum(np.power((dsource - dinput), 2))))

    def w_mj_vote(self, d_center):
        # 근접 이웃들을 구해온다
        dis_array = np.zeros(len(self.kdata))

        # 거리를 계산
        for i in range(0, len(self.kdata)):
            dis_array[i] = self.e_distance(d_center, self.kdata[i])

        ind_array = np.argsort(dis_array)
        near_array = ind_array[0:self.knei]

        # 이웃들의 가중치를 담은 배열
        w_array = np.zeros(self.knei)

        for i in range(0, len(near_array)):
            d_nei = self.kdata[int(near_array[i])]
            # exp(-x)로 가중치 계산
            w_array[i] = np.exp(-self.e_distance(d_center, d_nei))

        # Vote 값을 저장할 배열
        weightvote_array = np.zeros(10)

        # 근접 이웃을의 분류값을 이용해 투표
        for i in range(0, len(near_array)):
            # 각 투표의 중요도는 1 * 가중치로 계산한다.
            weightvote_array[int(self.klabel[near_array[i]])] += (1 * w_array[i])

        # 투표결과 가장 많은 결과의 반환
        return np.argsort(weightvote_array)[9]

    def __del__(self):
        # print('KNN class is removed Successfully!')
        pass


###################################################
