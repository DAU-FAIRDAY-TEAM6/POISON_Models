import torch
from transformers import pipeline
from transformers import BertModel, BertTokenizer
from tqdm import tqdm, trange,tnrange,tqdm_notebook


def get_Embedding_Using_BERT(marked_text): # marked_text = review
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained("bert-base-uncased") #model = BertModel.from_pretrained("bert-base-uncased", num_labels=4)
    
    
    # `encode_plus` 메서드 사용: max_length 설정 및 truncation 활성화
    encoded_dict = tokenizer.encode_plus(
        marked_text,                      # 입력 텍스트
        add_special_tokens=True,         # 특수 토큰 추가
        max_length=512,                  # 최대 길이 설정
        truncation=True,                 # 초과 시 잘라내기
        return_tensors='pt'              # PyTorch 텐서로 반환
    )
    
    tokens_tensor = encoded_dict['input_ids']
    segments_tensors = encoded_dict['token_type_ids']
    
    # tokenized_text = tokenizer.tokenize(marked_text) # text를 토큰으로 분리
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # 토큰화된 문자들을 숫자 리스트로 변경
    # segments_ids = [1] * len(tokenized_text) # 리뷰 하나를 통으로 넣기 때문에 단일 문장으로 바라보고 세그먼트를 하나만 사용한다.
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[0]
        sentence_Embedding = hidden_states[:,0,:] # CLS Token Embedding
        return sentence_Embedding.numpy()

