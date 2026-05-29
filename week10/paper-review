## Abstract

- Transformer 구조는 NLP분야에서 높은 성능을 보였지만, Computer Vision 분야에서는 여전히 제한적.
- Transformer 구조를 이미지 인식 작업에 직접 적용하는 Vision Transformer(ViT)를 제안
- 기존에는 Transformer를 이미지에 적용할 때 CNN과 결합하거나, 일부를 대체하는 방식이 일반적이었음.
- ViT는 이러한 CNN의 의존성을 제거하고 이미지를 patch의 시퀀스로 간주하여 순수 Transformer로 처리하는 방식을 택함.

- 대규모 데이터셋으로 사전 훈련된 ViT는 Image Net, CIFAR-100, VTAB등 다양한 이미지 분류 벤치마크에서 뛰어난 결과 달성, CNN 기반 모델에 비해 학습에 필요한 계산 자원을 상당히 적게 사용. → 대규모 데이터로 학습했을 때 맹우 강력해짐
- 이미지를 더 작은 고정 크기의 patch로 분할하고, 각 patch를 선형적으로 임베딩하여 시퀀스로 만든 후, 이를 표준 Transformer Encoder에 입력함.

<aside>
💡

- 이미지를 patch들의 시퀀스로 바꿔서 NLP의 Transformer처럼 처리하면, 충분히 큰 데이터셋에서 사전학습했을 때 CNN 없이도 이미지 분류에서 매우 강력한 성능을 낼 수 있다!
- NLP애서는 문장을 단어들의 시퀀스로 처리. 이미지도 비슷하게 16*16 patch로 쪼개고, 각 patch를 단어처럼 여겨서 Transformer에 넣음.
</aside>

## 1. Introduction

- NLP애서는 Self-attention 기반 Transformer가 지배적, 반면 Vision에서는 여전히 CNN이 중심.
- 이미지에는 지역성, 이동 불변성 같은 특성이 중요해서 CNN이 유리했기 때문
- CNN의 강한 inductive bias 없이도, 데이터가 충분하면 Transformer가 이미지를 잘 이해할 수 있음 → 모델 구조에 미리 심어놓은 가정보다, 대규모 데이터에서 학습되는 표현력을 더 믿는 방향

## 2. Related Work

ViT가 어디에서 왔는지?

### (1) CNN 기반 비전 모델

- 이미지 분류의 기본은 CNN 계열이었다.
- CNN은 local receptive field, weight sharing 같은 구조 덕분에 이미지 처리에 강하다.

### (2) Self-attention을 비전에 적용한 기존 연구

- 이전에도 attention을 비전에 쓰려는 시도는 있었다.
- 하지만 대부분은 CNN과 attention을 **섞어서** 사용했다.
- 또는 이미지 전체에 Transformer를 직접 적용하기엔 계산량이 너무 컸다.

### (3) NLP의 Transformer

- Transformer는 sequence modeling에 강하다.
- BERT 등에서 [CLS] token, positional embedding 같은 개념이 널리 쓰였다.
- 저자들은 이런 NLP용 Transformer를 이미지에도 거의 그대로 가져오려 한다.

<aside>
💡

ViT는 이미지를 patch 시퀀스로 바꾸고, 복잡한 CNN 보조장치 없이 거의 순수한 Transformer encoder만으로 강력한 성능을 보여줌

</aside>

## 3. Method

- 이미지를 패치로 분할
- 각 패치를 embedding
- positional embedding 추가
- [CLS] token 붙임
- Transformer encoder 통과
- 분류 헤드로 예측

### 3.1 Vision Transformer

표준 Transformer는 1차원(1D) 토큰 임베딩 시퀀스를 입력으로 받음. 

ViT는 이미지 $x∈R^{H×W×C}$를 $P×P$ 크기의 패치로 나눈다.

이렇게 나뉜 패치들은 펼쳐져(flattend) $N×(P2⋅C)$ 크기의 시퀀스 $x_p$ 가 됨. 

여기서  $N = HW/P^2$는 총 패치 수 나타냄, 이 값이 Transformer의 유효 입력 시퀀스 길이가 됨.

- 예시
    - 이미지 크기: $224×224$
    - 패치 크기: $16×16$
    
    그러면 한 변에 $224/16=14$개 패치가 생기고,
    
    총 패치 수는 $14×14=196$개
    
    각 패치는
    
    - 크기: $16×16×3$
    - 이를 일렬로 펴면 하나의 벡터가 됨
    
    즉 이미지 한 장이 **196개의 토큰 시퀀스**로 변하는 것.
    

이미지를 더 이상 “픽셀 격자”로 보는 게 아니라 “패치들의 나열”로 본다는 점이 핵심.

### 3.2 FIne Tuning and Higher Resolution

각 패치를 그냥 넣는 게 아니라, **선형변환**으로 D-차원 임베딩으로 바꿈.

- 사전 훈련 후 미세 조정: ViT모델을 대규모 데이터셋으로 사전 훈련한 후, 더 작은 규모의 하위 작업에 맞게 미세 조정(fine-tuning) 하는 것 → 사전 훈련된 예측 헤드를 제거하고, 하위 작업의 클래스 수(K)에 맞는 $D×K$ 크기의 0으로 초기화된 feedforward layer를 새로 추가함.

이미지 패치를 embedding vector로 바꾸는 것.

- flatten된 패치 벡터에
- 학습 가능한 projection matrix를 곱해서
- transformer 입력 차원 $D$ 로 맞춤

- Hybrid Architecture: 순수 ViT만 실험한 게 아니라, CNN backbone의 feature map을 패치처럼 취급하는 hybrid model도 실험함
