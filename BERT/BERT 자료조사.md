# BERT

## **Measuring Similarity of Opinion-bearing Sentences** 논문
    
### 모델

![image](https://github.com/choibumku00/POISON_Models/assets/101037541/be5b8d34-c5c6-4866-b51a-c3e43ca80d2f)


> 바닐라 모델 사진
    
![image](https://github.com/choibumku00/POISON_Models/assets/101037541/2c64a507-c2f6-42c3-ae55-cf8179ca74a4)

    

바닐라 모델의 SBERT는 2개의 BERT를 통해 문장간 유사도나 클래스 분류를 진행한다.

이 논문에서는 SBERT도 좋지만 triplet loss를 사용해서 긍정 리뷰, 부정 리뷰를 같이 사용해서 loss를 줄여나가며 BERT를 학습시킨다.

## 데이터 셋

![image](https://github.com/choibumku00/POISON_Models/assets/101037541/c6def358-8888-4df0-9007-23a57d413a21)

    
데이터셋은 SemEval 2016 Task 5의 데이터를 기반으로 합니다: "Aspect-Based Sentiment Analysis Subtask 1" (Pontiki 등, 2016). SemEval 데이터셋에는 영어로 된 노트북과 레스토랑에 대한 리뷰 문장이 포함되어 있습니다.

우리는 동일한 제품 또는 사업에 대한 리뷰에서 두 문장을 선택하여 문장 쌍을 생성하고, 이에 대한 인간의 유사성 판단을 수집했습니다.

우리의 데이터셋은 두 도메인만을 다루지만, 두 도메인 모두 일반적으로 사용되거나 사용 가능한 리뷰 데이터셋과 밀접한 관련이 있습니다. Yelp 리뷰 (Chu와 Liu, 2019; Bražinskas 등, 2020a)는 주로 레스토랑에 대한 것이고, 아마존의 전자제품 리뷰 (Angelidis와 Lapata, 2018; Bražinskas 등, 2020a)는 노트북 도메인과 밀접한 관련이 있습니다.

평가가 제품의 서로 다른 특징에 대해 간단히 이루어지지 않도록하기 위해 문장 사이에서 최소한 하나의 측면을 동일하게 유지했습니다. 이렇게 함으로써 주석은 평가의 표현에 따라 달라질 것입니다.

인간의 판단은 아마존 미케닉턱(Mechanical Turk)을 사용하여 수집되었습니다. "Mechanical Turk Masters Qualification"을 가진 주석 달기만 고려되었습니다. 주석 달기에 참여한 사람들은 각 의견 문장 쌍의 의미적 유사성을 5단계 리커트 척도를 사용하여 평가했습니다.

이 방법론은 Semantic Textual Similarity task (STS) 공유 작업에서 빌린 것입니다 (Cer 등, 2017). 우리의 주석 작업에서 척도는 0 ("전혀 다른 의견")부터 4 ("전적으로 같은 의견")까지이며, 중간 값인 2는 부분 일치를 나타냅니다.

우리는 주석을 세 가지 품질 기준에 따라 처리했습니다: (1) 품질 제어 문장에 대한 저 정확도 주석 제외; (2) 이상한 주석자 식별 및 제외; 그리고 (3) 주석자를 필터링 한 후 문장 쌍 당 최소 세 개의 주석 필요.

온라인 리뷰에는 일반적으로 리뷰 텍스트와 리뷰 평가가 포함됩니다. 리뷰 텍스트에는 의견 텍스트가 포함되어 있고 리뷰 평가는 리뷰 텍스트의 전반적인 감성 극성을 제공합니다. 인기 있는 리뷰 플랫폼에서는 리뷰 평가가 일반적으로 1부터 5까지의 점수 범위를 가집니다. 1의 리뷰 평가는 부정적이고 5는 긍정적입니다. 높은 등급의 리뷰 텍스트는 긍정적이라는 연결을 할 수 있습니다. 마찬가지로, 낮은 등급의 리뷰 텍스트는 부정적입니다. 저희의 작업에서, 우리는 긍정적인 리뷰를 4점과 5점으로 평가받은 리뷰로 간주하고, 부정적인 리뷰를 1점과 2점으로 평가받은 리뷰로 간주합니다. 우리는 3점의 등급을 가진 리뷰를 제외합니다. 동일한 제품에 대해, 같은 감성 극성을 가진 리뷰 텍스트(모두 긍정적이거나 모두 부정적인 경우)는 유사하다고 간주되고, 다른 감성 극성을 가진 리뷰 텍스트(하나는 긍정적이고 다른 하나는 부정적인 경우)는 다릅니다. 이것은 우리 모델을 세밀하게 조정하기 위한 교육 데이터셋을 만드는 기초가 됩니다.
의견 유사성 모델을 교육하기 위해 Siamese 네트워크와 Triplet 네트워크를 모두 탐색합니다. SOS, SOSS를 세밀하게 조정하기 위한 Siamese 네트워크는 의견 유사성을 적은 과제로 형식화하며, 학습 목표는 두 문장을 유사하거나 그렇지 않은 것으로 분류하는 것입니다. 지도는 쌍이 유사하거나 다른지의 이진 신호입니다. 이 접근 방식은 SBERT의 Siamese 네트워크가 SNLI와 MNLI 데이터셋에서 세밀하게 조정되는 문장 유사성을 위한 비지도 메트릭에서 사용됩니다. 이 작업에서, 우리는 리뷰 쌍의 교육, 개발 및 테스트 데이터셋을 생성합니다. 각 쌍은 동일한 제품의 리뷰를 포함합니다. 쌍은 유사하거나(모두 긍정적이거나 모두 부정적인) 다르게(하나는 긍정적이고 하나는 부정적인) 될 수 있습니다. 또한 데이터셋이 유사한 쌍과 다른 쌍이 균형을 이루도록 합니다. 교육 목표는 교차 엔트로피 손실입니다.

![image](https://github.com/choibumku00/POISON_Models/assets/101037541/628cba72-e5b8-4876-972e-826418ca8fc2)


## 추가 조사

위 모델도 좋지만 아래 사이트에도 yelp 리뷰 데이터 셋으로 학습 시킨 여러 BERT 모델이 있다.

성능은 뭐가 더 좋을지 모름

모델 찾기: https://huggingface.co/models?sort=downloads
    
- 모델 비교 (전부 yelp로 학습 시킨 것)

1. **[bert-base-uncased-yelp-polarity](https://huggingface.co/textattack/bert-base-uncased-yelp-polarity)** : [yelp_polarity](https://huggingface.co/datasets/yelp_polarity)로 학습된 모델 (리뷰 극성) 최대 토큰 256
2. **[YELP_BERT_5E](https://huggingface.co/pig4431/YELP_BERT_5E)** : [yelp_review_full](https://huggingface.co/datasets/yelp_review_full)로 학습된 모델(리뷰 전체 데이터)
3. [**yelp-longformer**](https://huggingface.co/Rveen/yelp-longformer): longformer는 최대 토큰 4096까지 받음
