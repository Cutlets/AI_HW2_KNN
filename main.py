# 홍익대학교 컴퓨터공학과 2021년 1학기 인공지능 HW2
# 학  번 : B511156
# 이  름 : 이 치 현

import sys
import os
import knn as kn
import numpy as np
import random as rd
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from tqdm import tqdm

# main.py
###################################################


def data_2xcomp(srcdata):
    tmpdata = []
    # 2픽셀마다 둘 사이의 평균을 구해 합쳐준다.
    for i in range(0, len(srcdata)//2):
        tmpdata.append((srcdata[i]+srcdata[i+1])/2)
    # 변환 결과를 쉬운 처리를 위해 사용했던 list에서 ndarray로 다시 바꾸어준다.
    cmpdata = np.asarray(tmpdata)
    return cmpdata


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True)
# x_train -> 이미지 데이터
# t_train -> 해당 이미지의 정답

# 이웃 수 설정
set_nei = 11
# 사용할 TEST 데이터 갯수
num_of_test = 10
# 정확도 측정
nhit = 0
accur = 0
# input 타입 0 : 모두 사용 /  나머지 : 압축 사용
input_type = 1

if input_type == 0:
  # 출력 양식 설명
  print('<n>th Data : ' + '\033[95m' + '<REAL>' + '\033[0m' + '|' + '\033[96m' + '<RESULT>' + '\033[0m')
  sk = kn.SimKnn(x_train, t_train, set_nei)

  # 지정된 횟수만큼 반복해준다
  for k in range(0, num_of_test):
      i = rd.randint(0, 10000)
      realValue = t_test[i]
      resultValue = sk.w_mj_vote(x_test[i])

      # 정답히면 nhit 증가
      if realValue == resultValue:
          nhit += 1
      print('%5s' % str(i) + ' th Data >> ' + '\033[95m' + str(realValue) + '\033[0m' + '|'\
              + '\033[96m' + str(resultValue) + '\033[0m')
else:
  # 훈련 데이터를 전처리
  tmp_train = []
  for i in tqdm(range(0, len(x_train))):
      cmp_set = data_2xcomp(x_train[i])
      tmp_train.append(cmp_set)
  c_train = np.asarray(tmp_train)

  sk = kn.SimKnn(c_train, t_train, set_nei)
  print('<n>th Data : ' + '\033[95m' + '<REAL>' + '\033[0m' + '|' + '\033[96m' + '<RESULT>' + '\033[0m')
  for k in range(0, num_of_test):
      i = rd.randint(0, 10000)
      realValue = t_test[i]
      resultValue = sk.w_mj_vote(data_2xcomp(x_test[i]))
      if realValue == resultValue:
          nhit += 1
      print('%5s' % str(i) + ' th Data >> ' + '\033[95m' + str(realValue) + '\033[0m' + '|'\
              + '\033[96m' + str(resultValue) + '\033[0m')

print('===')
accur = round(nhit/num_of_test, 4) * 100
print('Accuracy = ' + str(accur) + '%')


###################################################
