import torch
import torch.nn as nn
import torch.nn.functional as F

class VBPR(nn.Module):
    def __init__(self, num_users, num_items, num_visual_features, embedding_dim=20, visual_embedding_dim=10):
        super(VBPR, self).__init__()
        # 기본 BPR
        self.user_embedding = nn.Embedding(num_users, embedding_dim) # user 초기 임베딩 생성
        self.item_embedding = nn.Embedding(num_items, embedding_dim) # poi 초기 임베딩 생성
        
        # VBPR
        # 이미지는 선형 변환을 통해 차원 축소
        self.visual_item_embedding = nn.Linear(num_visual_features, visual_embedding_dim, bias=False) # 이미지 임베딩을 받아와 저차원으로 차원 축소
        self.user_visual_embedding = nn.Embedding(num_users, visual_embedding_dim) # 이미지 부분에 대한 user 임베딩 생성

    def forward(self, user_indices, item_pos_indices, item_neg_indices, visual_features_pos, visual_features_neg):
        # user, poi(긍정, 부정)에 대한 임베딩
        user_emb = self.user_embedding(user_indices)
        item_pos_emb = self.item_embedding(item_pos_indices)
        item_neg_emb = self.item_embedding(item_neg_indices)
        visual_pos_emb = self.visual_item_embedding(visual_features_pos)
        visual_neg_emb = self.visual_item_embedding(visual_features_neg)
        user_visual_emb = self.user_visual_embedding(user_indices)
        
        # 요소곱 후, 행(axis=1) 합산
        # 설명: 일반적인 MF의 내적과 달리 하나의 user가 하나의 poi에 대해 갈지 안갈지만 판단 
        # 요소곱 후, 합산하는 계산과 일반적인 MF에서의 내적 계산과 동일한 연산이다.
        # 기본 BPR과 VBPR은 단순히 더해줌
        pos_scores = (user_emb * item_pos_emb).sum(1) + (user_visual_emb * visual_pos_emb).sum(1)
        neg_scores = (user_emb * item_neg_emb).sum(1) + (user_visual_emb * visual_neg_emb).sum(1)
        
        return pos_scores, neg_scores # 추후 pos는 크게, neg는 작게 학습

    def bpr_loss(self, pos_scores, neg_scores):
        # BPR loss calculation using log sigmoid function
        # pos_scores가 크고, neg_scores는 작아야지 loss가 작음
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

# 샘플 예제
num_users = 1000
num_items = 500
num_visual_features = 2048
model = VBPR(num_users, num_items, num_visual_features)

# Dummy data 생성
user_indices = torch.randint(0, num_users, (10,))
item_pos_indices = torch.randint(0, num_items, (10,))
item_neg_indices = torch.randint(0, num_items, (10,))
visual_features_pos = torch.randn(10, num_visual_features)
visual_features_neg = torch.randn(10, num_visual_features)

# Forward
pos_scores, neg_scores = model(user_indices, item_pos_indices, item_neg_indices, visual_features_pos, visual_features_neg)

# BPR loss 출력
loss = model.bpr_loss(pos_scores, neg_scores)
print("Loss:", loss.item())
