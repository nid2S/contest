# why
- [데이콘 대회 링크](https://dacon.io/competitions/official/235747/overview/description)
- 해커톤을 위한 데이콘 뉴스토픽 분류 대회
- 뉴스 헤드라인을 이용, 뉴스의 주제 분류
# how
- 헤드라인의 정보를 압축(1DCNN/S2S인코더/트랜스포머인코더) > 전결합층(출력층, softmax) > 예측
- default - sparse_cross_entropy(Loss), accuracy/F1-measure(score), adam(optimizer)
- 그리드서치를 이용, 하이퍼파라미터(lr, emb_dim, batch_size, dropout_rate, etc), 모델구조의 최적값을 찾아냄
  (과정중 랜덤시드 고정(가중치 초기화?))
# notice
- pytocth/tensorflow - 텐서플로우로 제작 후 파이토치로 변경.
- S2S
