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
import Data_Preprocess as dp


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
        self.train_data = dataset.train
        self.test_data = dataset.test
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.neg = dataset.neg
        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.init_mean = init_mean
        self.init_stdev = init_stdev

        self.items_of_user = dataset.history_list

        # 사용자와 아이템의 잠재 요인을 초기화합니다.
        self.embed_user = torch.normal(mean=self.init_mean * torch.ones(self.num_user, self.factors), std=self.init_stdev).requires_grad_()
        self.embed_item = torch.normal(mean=self.init_mean * torch.ones(self.num_item, self.factors), std=self.init_stdev).requires_grad_()
        
        # Adam optimizer를 초기화합니다.
        self.mf_optim = optim.Adam([self.embed_user, self.embed_item], lr=self.learning_rate)

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
        y_ui = torch.mm(self.embed_user[u], self.embed_item[i].t()).sum(dim=-1)
        # 사용자와 부정적인 아이템 간의 예측 점수 계산
        y_uj = torch.mm(self.embed_user[u], self.embed_item[j].t()).sum(dim=-1)
        # 정규화 항 계산
        regularizer = self.reg * (torch.sum(self.embed_user[u] ** 2) + torch.sum(self.embed_item[i] ** 2) + torch.sum(self.embed_item[j] ** 2))
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
        
        for epoc in range(epoch):
            iter_loss = 0
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
            if epoc % 5 == 0:
                hits, recall = evaluate_model(self, self.test_data, topK)
            
                print("epoch=%d, loss = %.5f [%.1f s] HitRatio@%d = %.8f, RECAll@%d = %.8f [%.1f s]"
                        % (epoc, iter_loss, (t2 - t1), topK, hits, topK, recall, time.time() - t2))
                t1 = time.time()


    def predict(self, u, i):
        '''
        사용자와 아이템 사이의 점수를 예측합니다.
        Args:
            u: 사용자 ID.
            i: 아이템 ID.
        Returns:
            score: 사용자와 아이템 사이의 예측 점수.
        '''
        return np.inner(self.embed_user[u].detach().numpy(), self.embed_item[i].detach().numpy())

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

class Yelp(Dataset):
    def __init__(self):
        """
        Yelp 데이터셋을 로드하고 학습 데이터와 테스트 데이터를 생성합니다.

        Args:
            dir (str): 데이터 파일이 있는 디렉토리 경로.
            splitter (str): 파일에서 열을 구분하는 구분자.
            K (int): K 값, 즉 각 사용자마다 테스트에 사용되는 상호작용의 수.
        """
        path = 'Ours/dataset/'
        user_history_list, _,_,_,_,_ = dp.get_data(path)

        self.train = [] 
        self.test = []
        self.history_list = user_history_list
        
        self.num_user = len(user_history_list)
        self.num_item = 14586

        items = [i for i in range(self.num_item)]
        self.neg = dict()
        
        for u, hist in enumerate(user_history_list):
            random.shuffle(hist)
            self.train.append(hist[:int(len(hist) * 0.7)])
            self.test.append(hist[int(len(hist) * 0.7) :])
            
            u_negs = set(items) - set(hist) 
            self.neg[u] = list(u_negs) # ng dataset 생성
        
        self.test_for_eval = []
        for u,hist in enumerate(self.test):
            for i in hist:
                self.test_for_eval.append([u,i])

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
        j = self.neg[u][np.random.randint(0, len(self.neg[u]))]
        return (u, i, j)

def evaluate_model(model, test, K):
    """
    Top-K 추천의 성능(Hit_Ratio, NDCG)을 평가합니다.
    반환값: 각 테스트 상호작용의 점수.
    """
    score_matrix = torch.mm(model.embed_user, model.embed_item.t())
    top_scores, top_indicies = torch.topk(score_matrix, K, dim=1)
    
    hits = 0
    sum_recall = 0
    
    for u,hist in enumerate(test):
        set_topk = set(i.item() for i in (top_indicies[u]))
        set_hist = set(hist)
        
        if set_hist & set_topk:
            hits += 1
            
        sum_recall += len(set_hist & set_topk) / float(len(set_hist))
        
    return hits / len(test), sum_recall / len(test)

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
    args.batch_size = 32
    args.learning_rate = 0.01
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = "POISON_Models\BPR_Model\Ours\data/"  # 데이터셋 디렉토리 또는 파일
    yelp = Yelp()

    factors = 32  # 잠재요인 수
    learning_rate = args.learning_rate  # 학습률
    reg = 1e-5  # 정규화 계수
    init_mean = 0  # 초기 가중치 평균
    init_stdev = 0.01  # 초기 가중치 표준편차
    epoch = 50  # 최대 반복 횟수
    batch_size = args.batch_size  # 미니배치 크기
    num_thread = mp.cpu_count()  # 사용할 스레드 수
    K = 10
    print("#factors: %d, lr: %f, reg: %f, batch_size: %d" % (factors, learning_rate, reg, batch_size))

    # MF-BPR 모델 생성 및 학습
    bpr = MFbpr(yelp, factors, learning_rate, reg, init_mean, init_stdev)
    bpr.build_model(epoch, num_thread, batch_size=batch_size, topK = K)

    # 학습된 가중치 저장
    np.save("out/u"+str(learning_rate)+".npy", bpr.U.detach().numpy())
    np.save("out/v"+str(learning_rate)+".npy", bpr.V.detach().numpy())
