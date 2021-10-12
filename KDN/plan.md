# task
1. 문장 감성분류 : 문장(문서)을 입력받아 해당 문장의 감성을 확률과 함께 반환.
   - 주의 : 나눠질 감성의 분류/종류. 이미 있는 데이터를 기반으로 하나 정 없으면 라벨링.
2. 문법 오류 교정(Grammatical Error Correction, GEC) : 전체 문서를 입력받아 해당 문서에서 문법상 오류가 있는 부분을 알려주고/고침.
   - 주의 : API가 다수 존재함. 사용가능여부를 보고 사용결정, 활성화된 분야가 아니라 예시가 희박함. 논문보고 직접 구현해야 될 수 도 있을 듯.

# Emotion Classification
- kobert와 감성대화 말뭉치 데이터셋을 이용해 전처리 > 모델사용 > 출력층(softmax)의 과정을 거쳐 문장의 감성을 분류함.
- 출력층을 두개로 민들어, 6개의 대분류(분노/슬픔/불안/상처/당황/기쁨)로 하나, 60개의 소분류(대분류 6개+하위분류 각 9개씩)를 대분류+소분류 형태로 하나씩 각각 분류시킴.

# data
- [AI hub 감성 대화 말뭉치](https://aihub.or.kr/aidata/7978)

# problems
1. transfomer bert모델에서 tensorflow layer로 넘어가질 못함 | model.submodules 확인결과 수많은 tf레이어 + transformer레이어로 구성되어있음.
   - 기존 tf모델에 레이어 추가 방법 확인 / 레이어 분해 방법 확인 | Input(input_ids만) > bert 가 가능한지 확인 | classialAPI로 반환값 > Dense가 가능한지 확인

2. mobile_bert일 때 fit : `Shape must be rank 3 but is rank 4 for '{{node tf_mobile_bert_model/mobilebert/embeddings/Pad}} = 4
    Pad[T=DT_FLOAT, Tpaddings=DT_INT32](tf_mobile_bert_model/mobilebert/embeddings/strided_slice, 
    tf_mobile_bert_model/mobilebert/embeddings/Pad/paddings)' with input shapes: [?,88963,100,768], [3,2].`
   - 결국 데이터셋(from_tensors)이 문제였음이 발견되어 수정함(from_tensor_slices).
