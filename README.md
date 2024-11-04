# 컨텐츠 정보를 활용한 개인화 POI(Point-Of-Interest) 추천 시스템 개발
POISON 팀: 박동욱, 김진수, 김태희, 최범규  

## 연구 배경
기존 연구에서 POI추천에 있어 다양한 컨텐츠 정보를 충분히 활용하지 못했다. 본 연구에서는 리뷰 데이터와 지리정보를 활용하여 POI 추천을 하고자 한다.  
![image](https://github.com/user-attachments/assets/68b0d3ba-1333-42b3-a5cc-c8496181f5a9)

## 모델 파이프라인
### LightGCN, BERT
베이스 추천 모델로 LightGCN 사용  
리뷰 임베딩 추출로 BERT 사용  
![image](https://github.com/user-attachments/assets/b6d27e1b-b21b-4808-a1db-d7ee636f7618)  
### Distance BPR
BPR에 POI간 거리정보를 반영하기 위해 DistanceBPR 사용  
![image](https://github.com/user-attachments/assets/d484fdb5-4148-4e98-976a-e8469d3aa39d)

## 학습 데이터
POI 데이터 셋인 **Yelp** 데이터셋 사용
### 전처리 조건
![image](https://github.com/user-attachments/assets/6c5f5f54-c31a-4c20-b9ca-ad6c33d8b1d5)

위 조건으로 전처리를 진행하여, **37,685명의 사용자**가 **14,585개의 POI**에 남긴 **500,000여개의 리뷰 데이터**를 획득

## 실험 결과
![image](https://github.com/user-attachments/assets/ba81c86a-3709-416a-9e2a-0606d6368433)
