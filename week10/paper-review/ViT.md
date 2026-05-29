# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## Abstract

- Transformer 구조는 NLP 분야에서 높은 성능을 보였지만, Computer Vision 분야에서는 여전히 제한적으로 사용되었다.
- 본 논문은 Transformer 구조를 이미지 인식 작업에 직접 적용하는 **Vision Transformer (ViT)** 를 제안한다.
- 기존에는 Transformer를 이미지에 적용할 때 CNN과 결합하거나, CNN의 일부를 대체하는 방식이 일반적이었다.
- ViT는 이러한 CNN 의존성을 제거하고, 이미지를 **patch의 시퀀스**로 간주하여 순수 Transformer로 처리하는 방식을 택한다.
- 대규모 데이터셋으로 사전 훈련된 ViT는 ImageNet, CIFAR-100, VTAB 등 다양한 이미지 분류 벤치마크에서 뛰어난 결과를 달성했다.
- 또한 CNN 기반 모델에 비해 학습에 필요한 계산 자원을 상당히 적게 사용한다.
- 이미지를 더 작은 고정 크기의 patch로 분할하고, 각 patch를 선형적으로 임베딩하여 시퀀스로 만든 후, 이를 표준 Transformer Encoder에 입력한다.

> [!NOTE]
> 이미지를 patch들의 시퀀스로 바꾸어 NLP의 Transformer처럼 처리하면, 충분히 큰 데이터셋에서 사전 학습했을 때 CNN 없이도 이미지 분류에서 강력한 성능을 낼 수 있다.
>
> NLP에서는 문장을 단어들의 시퀀스로 처리한다. ViT는 이미지도 비슷하게 `16 × 16` 크기의 patch로 쪼개고, 각 patch를 하나의 단어처럼 여겨 Transformer에 입력한다.

---

## 1. Introduction

- NLP 분야에서는 Self-Attention 기반 Transformer가 지배적인 구조로 자리 잡았다.
- 반면 Computer Vision 분야에서는 여전히 CNN이 중심적인 모델 구조로 사용되고 있었다.
- 이미지에는 **지역성(locality)**, **이동 불변성(translation invariance)** 과 같은 특성이 중요하기 때문에 CNN이 유리했다.
- 하지만 ViT는 CNN의 강한 inductive bias 없이도, 데이터가 충분하다면 Transformer가 이미지를 잘 이해할 수 있음을 보여준다.
- 즉, 모델 구조에 미리 심어 놓은 가정보다 **대규모 데이터에서 학습되는 표현력**을 더 믿는 방향이다.

---

## 2. Related Work

ViT가 어떤 연구 흐름에서 등장했는지 정리하면 다음과 같다.

### 2.1 CNN-Based Vision Models

- 이미지 분류의 기본 모델은 CNN 계열이었다.
- CNN은 다음과 같은 구조적 특징 덕분에 이미지 처리에 강하다.
  - Local receptive field
  - Weight sharing
  - Translation equivariance

### 2.2 Self-Attention in Vision

- 이전에도 attention을 Computer Vision에 적용하려는 시도는 존재했다.
- 하지만 대부분은 CNN과 attention을 함께 사용하는 방식이었다.
- 또는 이미지 전체에 Transformer를 직접 적용하기에는 계산량이 너무 컸다.

### 2.3 Transformer in NLP

- Transformer는 sequence modeling에 강한 구조이다.
- BERT 등에서 `[CLS]` token, positional embedding 같은 개념이 널리 사용되었다.
- ViT의 저자들은 이러한 NLP용 Transformer 구조를 이미지에도 거의 그대로 가져오고자 했다.

> [!NOTE]
> ViT는 이미지를 patch 시퀀스로 바꾸고, 복잡한 CNN 보조 구조 없이 거의 순수한 Transformer Encoder만으로 강력한 성능을 보여준다.

---

## 3. Method

ViT의 전체 흐름은 다음과 같다.

1. 이미지를 patch로 분할한다.
2. 각 patch를 embedding vector로 변환한다.
3. Positional embedding을 추가한다.
4. `[CLS]` token을 붙인다.
5. Transformer Encoder를 통과시킨다.
6. Classification head를 통해 최종 클래스를 예측한다.

---

## 3.1 Vision Transformer

표준 Transformer는 1차원 토큰 임베딩 시퀀스를 입력으로 받는다.

ViT는 이미지 $x \in \mathbb{R}^{H \times W \times C}$를 $P \times P$ 크기의 patch로 나눈다.

나뉜 patch들은 펼쳐져(flattened) 다음과 같은 형태의 시퀀스가 된다.

$x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$

여기서 $N$은 전체 patch의 개수이며, Transformer의 유효 입력 시퀀스 길이가 된다.

$N = \frac{HW}{P^2}$

### Example

이미지 크기가 `224 × 224`이고, patch 크기가 `16 × 16`이라고 하자.

- 한 변에 생기는 patch 개수:

$224 / 16 = 14$

- 전체 patch 개수:

$14 \times 14 = 196$

- 각 patch의 크기:

$16 \times 16 \times 3$

각 patch를 일렬로 펼치면 하나의 벡터가 된다.  
즉, 이미지 한 장이 **196개의 토큰으로 이루어진 시퀀스**로 변환된다.

> [!IMPORTANT]
> ViT의 핵심은 이미지를 더 이상 픽셀 격자로만 보는 것이 아니라, **patch들의 나열**, 즉 sequence로 본다는 점이다.

---

## 3.2 Fine-Tuning and Higher Resolution

ViT에서는 각 patch를 그대로 Transformer에 넣지 않고, 먼저 **선형 변환(linear projection)** 을 통해 $D$차원 embedding vector로 변환한다.

즉, image patch를 embedding vector로 바꾸는 과정은 다음과 같다.

1. Patch를 flatten한다.
2. Flatten된 patch vector에 학습 가능한 projection matrix를 곱한다.
3. Transformer 입력 차원인 $D$에 맞춘다.

### Fine-Tuning

- ViT 모델은 먼저 대규모 데이터셋으로 사전 훈련된다.
- 이후 더 작은 규모의 downstream task에 맞게 fine-tuning을 수행한다.
- Fine-tuning 시에는 사전 훈련된 prediction head를 제거한다.
- 이후 하위 작업의 클래스 수 $K$에 맞는 $D \times K$ 크기의 feedforward layer를 새로 추가한다.
- 이 feedforward layer는 0으로 초기화된다.

### Higher Resolution

- Fine-tuning 단계에서는 사전 훈련 때보다 더 높은 해상도의 이미지를 사용할 수 있다.
- 이미지 해상도가 커지면 patch의 개수도 증가한다.
- 따라서 positional embedding의 크기도 조정이 필요하다.

### Hybrid Architecture

- 논문에서는 순수 ViT뿐만 아니라 hybrid architecture도 함께 실험했다.
- Hybrid model은 CNN backbone의 feature map을 patch처럼 취급하여 Transformer에 입력하는 방식이다.
- 이를 통해 CNN의 feature extraction 능력과 Transformer의 sequence modeling 능력을 함께 활용할 수 있다.

---

## Key Takeaways

- ViT는 이미지를 patch 단위로 나누어 Transformer에 입력하는 모델이다.
- 각 patch는 NLP에서의 token과 비슷한 역할을 한다.
- CNN 없이도 대규모 데이터셋으로 사전 학습하면 강력한 이미지 분류 성능을 낼 수 있다.
- ViT는 CNN의 inductive bias에 의존하기보다, 대규모 데이터로부터 표현을 학습하는 방식을 선택한다.
- 핵심 아이디어는 이미지를 **patch sequence**로 바꾸어 표준 Transformer Encoder에 입력하는 것이다.
```
