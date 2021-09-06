# why
- [데이콘 대회 링크](https://dacon.io/competitions/official/235747/overview/description)
- 해커톤을 위한 데이콘 뉴스토픽 분류 대회
- 뉴스 헤드라인을 이용, 뉴스의 주제 분류
# how
- 헤드라인의 정보를 압축(1DCNN/S2S인코더(RNN, Many-to-one)/트랜스포머인코더) > 전결합층(출력층, softmax) > 예측
- default - sparse_cross_entropy(Loss), accuracy/F1-measure(score), adam(optimizer)
- 그리드서치를 이용, 하이퍼파라미터(lr, epoch, emb_dim, batch_size, dropout_rate, etc), 모델구조의 최적값을 찾아냄
  (과정중 랜덤시드 고정(가중치 초기화?))
- 뉴스 토픽의 특성상 표현되는 주제의 범위와 표현 방식이 넓어 단어단위 분리는 힘들 듯.
- tensorflow로 구현 후 pytorch로도 구현
# structure
- 입력([인덱스, title\]형태) > 출력([인덱스, 토픽\]형태).
- train : 전처리 > 모델[입력데이터 변환 > 압축(Layer, CNN등) > 출력벡터\] > argmax() > loss/optimizer(backward)
- test : 전처리 > 모델 > argmax() > 재가공([인덱스, 토픽\]형태. 여기에 타이틀도 띄워주면 좋을 듯)
- 전처리 : (글자단위로 분리(OOV문제 제거, 종성이 빈 경우 따로 토큰 추가) > 글자 집합 생성 > 정수 인코딩) > 패딩(가장 긴 길이로) > 원핫인코딩 | torchtext의 Field사용
- 모델 : 기본 틀(출력층)만 제작 후 각 모델과 하이퍼 파라미터를 교차 사용/검증
- 검증 : train_test분할 > (기본 구조가 같은 모델 제작 > grid_search(교차검증 포함) > 구조가 다른 모델에도 반복 > 최고 정확도 산출
# have to check(hyper params)
- RNN/CNN/Transformer(Attention)
- optimizer order
- batch_size
- S2S_Encoder layer_num
- CNN kernel num/size
- RNN/BiRNN
- Embedding/Non-Embedding(FastText)
- Layer_num(depth, output_layer/hidden_layer)
- embedding/hidden_dim
- dropout rate
- learnig rate
- weight initialization
- etc
