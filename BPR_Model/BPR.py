import time
import numpy as np
from numpy import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import multiprocessing as mp
import argparse

import math
import heapq # for retrieval topK

class MFbpr(nn.Module):
    '''
    MF 모델에 대한 BPR 학습
    '''
    def __init__(self, dataset, factors, learning_rate, reg, init_mean, init_stdev):
        '''
        생성자
        Args:
            dataset: 데이터셋 객체로, 학습 및 테스트 데이터를 포함합니다.
            factors (int): 잠재 요인의 수.
            learning_rate (float): 최적화에 사용되는 학습률.
            reg (float): 정규화 강도.
            init_mean (float): 초기화에 사용되는 정규 분포의 평균.
            init_stdev (float): 초기화에 사용되는 정규 분포의 표준 편차.
        '''
        super(MFbpr, self).__init__()
        self.dataset = dataset
        self.train = dataset.train
        self.test = dataset.test
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.neg = dataset.neg
        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.init_mean = init_mean
        self.init_stdev = init_stdev

        # 사용자와 아이템의 잠재 요인을 초기화합니다.
        self.U = torch.normal(mean=self.init_mean * torch.ones(self.num_user, self.factors), std=self.init_stdev).requires_grad_()
        self.V = torch.normal(mean=self.init_mean * torch.ones(self.num_item, self.factors), std=self.init_stdev).requires_grad_()

        # Adam optimizer를 초기화합니다.
        self.mf_optim = optim.Adam([self.U, self.V], lr=self.learning_rate)
        self.items_of_user = []
        self.num_rating = 0
        for u in range(len(self.train)):
            # 각 사용자가 평가한 아이템 목록을 저장합니다.
            self.items_of_user.append(set([]))
            for i in range(len(self.train[u])):
                item = self.train[u][i][0]
                self.items_of_user[u].add(item)
                self.num_rating += 1


    def forward(self, u, i, j):
        '''
        MF-BPR 모델의 forward pass입니다.
        Args:
            u: 사용자 ID.
            i: 긍정적인 아이템 ID.
            j: 부정적인 아이템 ID.
        Returns:
            y_ui: 사용자와 긍정적인 아이템 간의 예측 점수.
            y_uj: 사용자와 부정적인 아이템 간의 예측 점수.
            loss: BPR 손실.
        '''
        # 사용자와 긍정적인 아이템 간의 예측 점수 계산
        y_ui = torch.diag(torch.mm(self.U[u], self.V[i].t()))
        # 사용자와 부정적인 아이템 간의 예측 점수 계산
        y_uj = torch.diag(torch.mm(self.U[u], self.V[j].t()))
        # 정규화 항 계산
        regularizer = self.reg * (torch.sum(self.U[u] ** 2) + torch.sum(self.V[i] ** 2) + torch.sum(self.V[j] ** 2))
        # BPR 손실 계산
        loss = regularizer - torch.sum(torch.log2(torch.sigmoid(y_ui - y_uj)))
        return y_ui, y_uj, loss

    def build_model(self, epoch=30, num_thread=8, batch_size=32, topK = 10):
        '''
        MF-BPR 모델을 구축하고 학습합니다.
        Args:
            epoch (int): 학습의 최대 반복 횟수.
            num_thread (int): 병렬 실행을 위한 스레드 수.
            batch_size (int): 학습용 배치 크기.
        '''
        data_loader = DataLoader(self.dataset, batch_size=batch_size)

        print("Training MF-BPR with: learning_rate=%.4f, regularization=%.4f, factors=%d, #epoch=%d, batch_size=%d."
               % (self.learning_rate, self.reg, self.factors, epoch, batch_size))
        t1 = time.time()
        iter_loss = 0
        for epoc in range(epoch):
            for s, (users, items_pos, items_neg) in enumerate(data_loader):
                # 기울기 초기화
                self.mf_optim.zero_grad()
                # Forward pass를 통해 예측과 손실 계산
                y_ui, y_uj, loss = self.forward(users, items_pos, items_neg)
                iter_loss += loss
                # Backward pass 및 파라미터 업데이트
                loss.backward()
                self.mf_optim.step()
            t2 = time.time()
            
            # 성능 측정 함수를 통해 HitRatio 및 NDCG를 계산
            (hits, ndcgs) = evaluate_model(self, self.test, topK, num_thread)
            hr_mean = np.array(hits).mean()
            ndcg_mean = np.array(ndcgs).mean()

            print("epoch=%d [%.1f s] HitRatio@%d = %.4f, NDCG@%d = %.4f [%.1f s]"
                    % (epoc, (t2 - t1), topK, hr_mean, topK, ndcg_mean, time.time() - t2))
            t1 = time.time()
            iter_loss = 0


    def predict(self, u, i):
        '''
        사용자와 아이템 사이의 점수를 예측합니다.
        Args:
            u: 사용자 ID.
            i: 아이템 ID.
        Returns:
            score: 사용자와 아이템 사이의 예측 점수.
        '''
        return np.inner(self.U[u].detach().numpy(), self.V[i].detach().numpy())

    def get_batch(self, batch_size):
        '''
        학습 데이터의 배치를 가져옵니다.
        Args:
            batch_size (int): 배치 크기.
        Returns:
            users: 사용자 ID 목록.
            pos_items: 긍정적인 아이템 ID 목록.
            neg_items: 부정적인 아이템 ID 목록.
        '''
        users, pos_items, neg_items = [], [], []
        for i in range(batch_size):
            u = np.random.randint(0, self.num_user)
            i = self.train[u][np.random.randint(0, len(self.train[u]))][0]
            j = np.random.randint(0, self.num_item)
            while j in self.items_of_user[u]:
                j = np.random.randint(0, self.num_item)
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
        return (users, pos_items, neg_items)

def LoadRatingFile_HoldKOut(filename, splitter, K):
    """
    주어진 .rating 파일을 읽고 Hold-K-Out 교차 검증을 위한 학습 및 테스트 데이터를 생성합니다.

    Args:
        filename (str): .rating 파일의 경로.
        splitter (str): 파일에서 열을 구분하는 구분자.
        K (int): K 값, 즉 각 사용자마다 테스트에 사용되는 상호작용의 수.

    Returns:
        train (list): Hold-K-Out 교차 검증을 위한 학습 데이터.
        test (list): Hold-K-Out 교차 검증을 위한 테스트 데이터.
        num_user (int): 사용자 수.
        num_item (int): 아이템 수.
        num_ratings (int): 전체 상호작용 수.
    """
    train = []
    test = []

    num_ratings = 0
    num_item = 0
    # 파일을 읽어서 train 및 test 데이터 생성
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(splitter)
            if (len(arr) < 4):
                continue
            user, item, time = int(arr[0]), int(arr[1]), int(arr[3])
            # 사용자별로 상호작용 데이터를 저장
            if (len(train) <= user):
                train.append([])
            train[user].append([item, time])
            num_ratings += 1
            num_item = max(item, num_item)
            line = f.readline()
    num_user = len(train)
    num_item = num_item + 1

    # 상호작용 데이터를 시간순으로 정렬
    def getTime(item):
        return item[-1];
    for u in range (len(train)):
        train[u] = sorted(train[u], key = getTime)

    # Hold-K-Out 교차 검증을 위해 학습 및 테스트 데이터 생성
    for u in range (len(train)):
        for k in range(K):
            if (len(train[u]) == 0):
                break
            # 가장 최근에 발생한 상호작용을 테스트 데이터로 이동
            test.append([u, train[u][-1][0], train[u][-1][1]])
            del train[u][-1]

    test = sorted(test, key = getTime)

    return train, test, num_user, num_item, num_ratings

class Pinterest(Dataset):
    def __init__(self, dir, splitter, K):
        """
        Pinterest 데이터셋을 로드하고 학습 데이터와 테스트 데이터를 생성합니다.

        Args:
            dir (str): 데이터 파일이 있는 디렉토리 경로.
            splitter (str): 파일에서 열을 구분하는 구분자.
            K (int): K 값, 즉 각 사용자마다 테스트에 사용되는 상호작용의 수.
        """

        self.train = []

        self.num_ratings = 0
        self.num_item = 0
        # pos.txt 파일을 읽어서 학습 데이터 생성
        with open(dir+'pos.txt', "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(splitter)
                if (len(arr) < 2):
                    continue
                user, item = int(arr[0]), int(arr[1])
                # 사용자별로 상호작용 데이터를 저장
                if (len(self.train) <= user):
                    self.train.append([])
                self.train[user].append([item])
                self.num_ratings += 1
                self.num_item = max(item, self.num_item)
                line = f.readline()
        self.num_user = len(self.train)
        self.num_item = self.num_item + 1

        self.test = []
        self.neg = dict()
        user = 0
        # neg.txt 파일을 읽어서 테스트 데이터 및 부정적 상호작용 데이터 생성
        with open(dir+'neg.txt', 'r') as f_neg:
            line = f_neg.readline()
            while line != None and line != '':
                arr = line.split(splitter)
                pos = int(arr[0])
                # 테스트 데이터 생성
                self.test.append([user, pos])
                # 사용자별로 부정적 상호작용 데이터 저장
                self.neg[user] = []
                for neg_i in range(len(arr)):
                    if arr[neg_i] != '\n':
                        self.neg[user].append(int(arr[neg_i]))

                user += 1
                line = f_neg.readline()
        print("#users: %d, #items: %d, #ratings: %d" %(self.num_user, self.num_item, self.num_ratings))


    def __len__(self):
        """
        데이터셋의 사용자 수를 반환합니다.
        """
        return self.num_user


    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 가져옵니다.

        Args:
            idx (int): 데이터셋 내의 인덱스.

        Returns:
            u: 사용자 ID.
            i: 긍정적인 아이템 ID.
            j: 부정적인 아이템 ID.
        """
        u = idx
        # 사용자별로 하나의 긍정적인 상호작용 선택
        i = self.train[u][np.random.randint(0, len(self.train[u]))]
        # 부정적인 상호작용 무작위 선택
        j = np.random.randint(0, self.num_item)
        while j in self.train[u]:
            j = np.random.randint(0, self.num_item)
        return (u, i, j)

def evaluate_model(model, testRatings, K, num_thread):
    """
    Top-K 추천의 성능(Hit_Ratio, NDCG)을 평가합니다.
    반환값: 각 테스트 상호작용의 점수.
    """
    global _model
    global _testRatings
    global _K
    _model = model
    _testRatings = testRatings
    _K = K
    num_rating = len(testRatings)

    hits = []
    ndcgs = []
    
    for i in range(num_rating):
        res = eval_one_rating(i)
        hits.append(res[0])
        ndcgs.append(res[1])
    
    return (hits, ndcgs)
def eval_one_rating(idx):
    rating = _testRatings[idx]
    hr = ndcg = 0
    u = rating[0]
    gtItem = rating[1]
    map_item_score = {}

    maxScore = _model.predict(u, gtItem)

    countLarger = 0

    for i in _model.neg[u]:

        early_stop = False
        score = _model.predict(u, i)
        map_item_score[i] = score

        if score > maxScore:
            countLarger += 1
        if countLarger > _K:
            hr = ndcg = 0
            early_stop = True
            break

    if early_stop == False:
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)

    return (hr, ndcg)
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
def parse_args():
    """
    명령행 인자를 파싱합니다.
    """
    args = argparse.Namespace()
    args.batch_size = 64
    args.learning_rate = 0.001
    return args

if __name__ == '__main__':
    print(mp.cpu_count(), mp.current_process().name)
    
    args = parse_args()
    dataset = "POISON_Models\BPR_Model\data/"  # 데이터셋 디렉토리 또는 파일
    splitter = " "  # 데이터 구분자
    hold_k_out = 1  # Hold-K-Out 교차 검증의 K 값
    pinterest = Pinterest(dataset, splitter, hold_k_out)

    factors = 64  # 잠재요인 수
    learning_rate = args.learning_rate  # 학습률
    reg = 1e-5  # 정규화 계수
    init_mean = 0  # 초기 가중치 평균
    init_stdev = 0.01  # 초기 가중치 표준편차
    epoch = 30  # 최대 반복 횟수
    batch_size = args.batch_size  # 미니배치 크기
    num_thread = mp.cpu_count()  # 사용할 스레드 수
    K = 10
    print("#factors: %d, lr: %f, reg: %f, batch_size: %d" % (factors, learning_rate, reg, batch_size))

    # MF-BPR 모델 생성 및 학습
    bpr = MFbpr(pinterest, factors, learning_rate, reg, init_mean, init_stdev)
    bpr.build_model(epoch, num_thread, batch_size=batch_size, topK = K)

    # 학습된 가중치 저장
    np.save("out/u"+str(learning_rate)+".npy", bpr.U.detach().numpy())
    np.save("out/v"+str(learning_rate)+".npy", bpr.V.detach().numpy())
