# PyTorch 기초 정리

## 1장. PyTorch 개요

### PyTorch란?

**PyTorch**는 파이썬의 간결함과 수치 연산의 강력함을 결합한 딥러닝 프레임워크이다.

PyTorch는 텐서 연산, 자동 미분, 신경망 모델 구현, GPU 연산 등을 지원하여 딥러닝 모델을 쉽게 만들고 학습시킬 수 있다.

---

## 동적 계산 그래프

### 동적 계산 그래프란?

동적 계산 그래프는 **필요할 때마다 그때그때 만들어지는 임시 설계도**라고 이해할 수 있다.

즉, 모델이 데이터를 처리하는 과정에서 계산 그래프가 즉석에서 생성되기 때문에 코드 실행 중에 구조를 유연하게 바꿀 수 있다.

| 구분        | 설명                     | 대표 프레임워크   |
| --------- | ---------------------- | ---------- |
| 정적 계산 그래프 | 미리 계산 구조를 정의하고 한 번에 실행 | TensorFlow |
| 동적 계산 그래프 | 실행하면서 즉석에서 계산 구조를 생성   | PyTorch    |

### PyTorch의 특징

* 파이썬 문법과 비슷해서 코드 작성이 직관적이다.
* 동적 계산 그래프를 사용하여 디버깅이 쉽다.
* GPU 연산을 지원하여 대규모 딥러닝 학습에 활용할 수 있다.
* 연구와 실험에서 모델 구조를 유연하게 바꾸기 좋다.

---

# 2장. Tensor 기초

## 텐서란?

**텐서(Tensor)**는 다차원 배열을 일반화한 자료구조이다.

딥러닝에서는 이미지, 텍스트, 음성, 숫자 데이터 등을 텐서 형태로 저장하고 연산한다.

예를 들어 다음과 같이 이해할 수 있다.

| 차원        | 예시                        |
| --------- | ------------------------- |
| 0차원 텐서    | 스칼라 값, 예: `3`             |
| 1차원 텐서    | 벡터, 예: `[1, 2, 3]`        |
| 2차원 텐서    | 행렬, 예: `[[1, 2], [3, 4]]` |
| 3차원 이상 텐서 | 이미지 데이터, 배치 데이터 등         |

---

## 텐서의 핵심 속성

PyTorch 텐서는 주로 다음 세 가지 속성을 가진다.

| 속성        | 의미                        |
| --------- | ------------------------- |
| Shape     | 텐서의 모양, 즉 각 차원의 크기        |
| Data Type | 텐서 안에 들어 있는 데이터의 자료형      |
| Device    | 텐서가 저장된 위치, 예: CPU 또는 GPU |

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x.shape)
print(x.dtype)
print(x.device)
```

---

## 텐서 생성 방법

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])
```

### dtype 지정하기

`dtype`은 텐서 안에 들어갈 데이터의 자료형을 명확히 정해주는 역할을 한다.

```python
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = torch.tensor([1, 2, 3], dtype=torch.int64)
```

| dtype           | 의미                |
| --------------- | ----------------- |
| `torch.float32` | 32비트 실수형          |
| `torch.int64`   | 64비트 정수형, long 타입 |

### 자료형 변환

이미 만들어진 텐서의 자료형은 다음과 같이 바꿀 수 있다.

```python
x = torch.tensor([1, 2, 3])

x_float = x.float()
x_int = x.int()
x_long = x.long()
x_double = x.double()
```

| 코드          | 의미            |
| ----------- | ------------- |
| `.float()`  | 실수형으로 변환      |
| `.int()`    | 정수형으로 변환      |
| `.long()`   | 64비트 정수형으로 변환 |
| `.double()` | 64비트 실수형으로 변환 |

---

# 텐서 기본 연산

## 산술 연산

텐서끼리의 덧셈, 뺄셈, 곱셈, 나눗셈은 기본적으로 **원소별 연산**으로 이루어진다.

즉, 같은 위치에 있는 값들끼리 연산된다.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
```

출력 결과:

```python
tensor([5, 7, 9])
tensor([-3, -3, -3])
tensor([ 4, 10, 18])
tensor([0.2500, 0.4000, 0.5000])
```

원소별 곱셈과 나눗셈은 신호 정규화, 손실 계산 등 다양한 상황에서 활용된다.

---

## 브로드캐스팅

**브로드캐스팅(Broadcasting)**은 서로 다른 shape을 가진 텐서끼리도 연산할 수 있도록 해주는 규칙이다.

작은 텐서가 자동으로 확장되어 큰 텐서와 같은 모양처럼 연산된다.

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([10, 20, 30])

print(a + b)
```

출력 결과:

```python
tensor([[11, 22, 33],
        [14, 25, 36]])
```

---

## 인덱싱과 슬라이싱

2차원 텐서에서는 다음과 같이 특정 행과 열의 값을 선택할 수 있다.

```python
tensor2d[row_index, column_index]
```

예시:

```python
tensor2d = torch.tensor([[10, 20, 30],
                         [40, 50, 60],
                         [70, 80, 90]])

print(tensor2d[0, 1])
```

출력 결과:

```python
tensor(20)
```

`:` 기호를 사용하면 여러 행 또는 여러 열을 동시에 선택할 수 있다.

```python
print(tensor2d[:, 1])
```

출력 결과:

```python
tensor([20, 50, 80])
```

---

## 불리언 마스킹

**불리언 마스킹(Boolean Masking)**은 특정 조건을 만족하는 데이터만 선택하거나 변경할 때 사용한다.

### 마스크 텐서란?

마스크 텐서는 원래 텐서와 같은 shape을 가지며, 각 원소가 `True` 또는 `False` 값을 갖는 텐서이다.

보통 조건식을 적용해서 만든다.

```python
data = torch.tensor([[10, 20, 30],
                     [30, 40, 50],
                     [50, 60, 70]])

mask = data > 50
print(mask)
```

출력 결과:

```python
tensor([[False, False, False],
        [False, False, False],
        [False,  True,  True]])
```

마스크를 인덱스로 사용하면 조건에 맞는 원소만 선택할 수 있다.

```python
selected_data = data[mask]
print(selected_data)
```

출력 결과:

```python
tensor([60, 70])
```

조건에 맞는 값만 변경할 수도 있다.

```python
data[mask] = -1
print(data)
```

출력 결과:

```python
tensor([[10, 20, 30],
        [30, 40, 50],
        [50, -1, -1]])
```

불리언 마스킹은 데이터 필터링, 효율적인 연산, 동적 데이터 처리 등에 활용된다.

---

# 고급 텐서 조작

## 텐서 이어 붙이기와 쌓기

### `torch.cat`

`torch.cat`은 특정 차원을 기준으로 텐서를 이어 붙인다.

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

result = torch.cat([a, b], dim=0)
print(result)
```

출력 결과:

```python
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
```

---

### `torch.stack`

`torch.stack`은 새로운 차원을 만들어 텐서를 쌓는다.

```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])

result = torch.stack([a, b], dim=0)
print(result)
```

출력 결과:

```python
tensor([[1, 2],
        [3, 4]])
```

`cat`은 기존 차원에 이어 붙이고, `stack`은 새로운 차원을 만든다는 점이 중요하다.

---

## 텐서 나누기

### `torch.split`

`torch.split`은 특정 크기만큼 텐서를 잘라서 반환한다.

```python
x = torch.tensor([1, 2, 3, 4, 5, 6])

result = torch.split(x, 2)
print(result)
```

출력 결과:

```python
(tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))
```

---

### `torch.chunk`

`torch.chunk`는 텐서를 여러 개로 나눌 때 **개수 기준**으로 나눈다.

```python
x = torch.tensor([1, 2, 3, 4, 5, 6])

result = torch.chunk(x, 3)
print(result)
```

출력 결과:

```python
(tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))
```

---

## 차원 변경

### `transpose`와 `t()`

`transpose`는 차원을 서로 바꾸는 함수이다.

2차원 텐서에서는 행과 열을 뒤바꿀 수 있다.

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.t())
```

출력 결과:

```python
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

---

### `permute`

`permute`는 3차원 이상의 텐서에서 차원의 순서를 바꿀 때 사용한다.

```python
x = torch.randn(2, 3, 4)

result = x.permute(1, 0, 2)
print(result.shape)
```

출력 결과:

```python
torch.Size([3, 2, 4])
```

---

## 차원 추가와 제거

### `squeeze`

`squeeze`는 크기가 1인 차원을 제거한다.

```python
x = torch.randn(1, 3, 1, 4)

result = x.squeeze()
print(result.shape)
```

출력 결과:

```python
torch.Size([3, 4])
```

---

### `unsqueeze`

`unsqueeze`는 특정 위치에 크기가 1인 차원을 추가한다.

```python
x = torch.tensor([1, 2, 3])

result = x.unsqueeze(0)
print(result.shape)
```

출력 결과:

```python
torch.Size([1, 3])
```

---

# Tensor와 NumPy 배열 비교

PyTorch 텐서와 NumPy 배열은 모두 다차원 배열을 다룬다는 점에서 비슷하다.

하지만 PyTorch 텐서는 GPU 연산과 자동 미분을 지원한다는 점에서 딥러닝 학습에 더 적합하다.

| 구분     | NumPy 배열      | PyTorch Tensor |
| ------ | ------------- | -------------- |
| 기본 목적  | 수치 연산         | 딥러닝 연산         |
| GPU 연산 | 기본적으로 지원하지 않음 | 지원             |
| 자동 미분  | 지원하지 않음       | 지원             |
| 딥러닝 활용 | 제한적           | 매우 적합          |

---

# 3장. 신경망 기초

## 신경망의 종류

대표적인 신경망 구조에는 다음과 같은 것들이 있다.

| 신경망 | 설명                      | 주요 활용           |
| --- | ----------------------- | --------------- |
| MLP | 기본적인 완전연결 신경망           | 분류, 회귀          |
| CNN | 이미지의 공간적 특징을 잘 추출하는 신경망 | Computer Vision |
| RNN | 순서가 있는 데이터를 처리하는 신경망    | 자연어, 시계열        |

---

## 신경망 응용 분야

| 분야  | 설명                                  |
| --- | ----------------------------------- |
| CV  | Computer Vision, 이미지와 영상 처리         |
| NLP | Natural Language Processing, 자연어 처리 |

---

# MLP

## MLP란?

**MLP(Multi-Layer Perceptron)**는 여러 개의 층으로 이루어진 기본적인 신경망이다.

MLP는 다음과 같은 구조로 이루어진다.

1. Input Layer
2. Hidden Layer
3. Output Layer

```text
Input Layer → Hidden Layer → Output Layer
```

입력 데이터가 은닉층을 거쳐 출력층으로 전달되면서 예측값이 만들어진다.

---

## MLP 학습 과정에서 중요한 요소

### Loss Function

Loss Function은 모델의 예측값과 실제 정답 사이의 차이를 계산한다.

분류 문제에서는 **Cross-Entropy Loss**가 많이 사용된다.

### Optimizer

Optimizer는 손실값을 줄이기 위해 모델의 가중치를 업데이트하는 역할을 한다.

대표적으로 **SGD**, **Adam** 등이 있다.

### Validation Set

Validation Set은 모델이 학습 데이터에만 잘 맞는지, 새로운 데이터에도 잘 작동하는지 확인하기 위해 사용한다.

### Early Stopping

Early Stopping은 Validation 성능이 더 이상 좋아지지 않을 때 학습을 멈추는 방법이다.

이를 통해 Overfitting을 방지할 수 있다.

---

# Activation Function

## 활성화 함수란?

**Activation Function**은 신경망에 비선형성을 추가하는 함수이다.

활성화 함수가 없으면 여러 층을 쌓아도 단순한 선형 모델과 비슷해지기 때문에, 복잡한 패턴을 학습하기 어렵다.

---

## Sigmoid 함수

Sigmoid 함수는 출력값을 0과 1 사이로 변환한다.

```text
출력 범위: (0, 1)
```

### 장점

* 확률처럼 해석하기 쉽다.

### 단점

* 입력값이 너무 크거나 작으면 기울기가 거의 0이 된다.
* 이로 인해 기울기 소실 문제가 발생할 수 있다.

---

## ReLU 함수

ReLU는 입력이 0 이하이면 0을 출력하고, 양수이면 그대로 출력한다.

```text
ReLU(x) = max(0, x)
```

### 장점

* 계산이 간단하다.
* Sigmoid보다 기울기 소실 문제를 완화할 수 있다.

### 단점

* 입력이 계속 음수이면 뉴런이 거의 학습되지 않는 **Dying ReLU** 문제가 발생할 수 있다.

---

## Leaky ReLU

Leaky ReLU는 ReLU의 Dying ReLU 문제를 완화하기 위해 사용된다.

입력이 음수일 때도 아주 작은 기울기를 남겨둔다.

---

## Tanh 함수

Tanh 함수는 출력값을 -1과 1 사이로 변환한다.

```text
출력 범위: (-1, 1)
```

### 장점

* 출력 중심이 0이기 때문에 중심이 0인 데이터를 학습하는 데 유리하다.

### 단점

* 여전히 기울기 소실 문제가 발생할 수 있다.

---

# Optimizer

## SGD

**SGD(Stochastic Gradient Descent)**는 확률적 경사 하강법이다.

전체 데이터를 한 번에 모두 사용하지 않고, 반복마다 일부 샘플을 사용하여 가중치를 업데이트한다.

### 특징

* 계산량이 상대적으로 적다.
* 무작위 샘플을 사용하기 때문에 업데이트에 노이즈가 있다.
* 노이즈 덕분에 지역 최솟값에서 벗어날 가능성이 있지만, 학습이 불안정할 수 있다.

---

## Mini-Batch SGD

Mini-Batch SGD는 전체 데이터도 아니고 샘플 하나도 아닌, 작은 묶음 단위로 데이터를 나누어 학습하는 방법이다.

딥러닝에서는 Mini-Batch SGD가 많이 사용된다.

---

## Adam Optimizer

Adam은 SGD의 단점을 보완한 옵티마이저이다.

각 파라미터마다 학습률을 동적으로 조정하여 더 빠르고 안정적인 학습을 돕는다.

### 특징

* 학습 속도가 빠른 편이다.
* 학습률을 자동으로 조정한다.
* 딥러닝 모델에서 자주 사용된다.

---

# PyTorch 유용한 함수

## 모델 저장과 불러오기

### 모델 저장

```python
torch.save(model.state_dict(), "model.pth")
```

### 모델 불러오기

```python
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## NaN 확인

학습 중 손실값이 비정상적으로 커지면 `NaN`이 발생할 수 있다.

```python
torch.isnan(loss).any()
```

이 코드는 손실값에 NaN이 포함되어 있는지 확인할 때 사용한다.

---

## GPU 사용

GPU를 사용하면 딥러닝 모델 학습 속도를 높일 수 있다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
data = data.to(device)
```

---

## 데이터 증강

데이터 증강은 기존 데이터를 변형하여 학습 데이터를 늘리는 기법이다.

이미지 데이터에서는 회전, 뒤집기, 자르기, 밝기 조절 등이 사용된다.

데이터 증강을 사용하면 모델이 다양한 상황을 학습할 수 있어 일반화 성능을 높이는 데 도움이 된다.

---

# 예제 문제

## 문제

다음 코드를 실행했을 때 최종 `data` 텐서의 모양은 어떻게 되는가?

```python
import torch

data = torch.tensor([[10, 20, 30],
                     [30, 40, 50],
                     [50, 60, 70]])

tensor2d = torch.tensor([[10, 20, 30],
                         [30, 40, 50],
                         [50, 60, 70]])

mask = tensor2d > 30
print(f"mask:\n{mask}")
print(f"30보다 큰 값들: {tensor2d[mask]}")

mask = data > 50
print(f"{mask}")

selected_data = data[mask]
print(f"50보다 큰 값들: {selected_data}")

data[mask] = -1
print(f"{data}")
```

---

## 풀이

먼저 다음 조건을 적용한다.

```python
mask = data > 50
```

`data`에서 50보다 큰 값은 `60`, `70`이다.

따라서 마스크는 다음과 같다.

```python
tensor([[False, False, False],
        [False, False, False],
        [False,  True,  True]])
```

이후 다음 코드가 실행된다.

```python
data[mask] = -1
```

마스크가 `True`인 위치의 값만 `-1`로 바뀐다.

즉, `60`과 `70`이 `-1`로 변경된다.

---

## 정답

최종 `data` 텐서는 다음과 같다.

```python
tensor([[10, 20, 30],
        [30, 40, 50],
        [50, -1, -1]])
```

최종 텐서의 shape은 다음과 같다.

```python
torch.Size([3, 3])
```

---

# 핵심 요약

* PyTorch는 동적 계산 그래프를 사용하는 딥러닝 프레임워크이다.
* 텐서는 다차원 배열이며, 딥러닝 데이터 표현의 기본 단위이다.
* 텐서의 핵심 속성은 Shape, Data Type, Device이다.
* 텐서 연산은 기본적으로 원소별 연산으로 이루어진다.
* 브로드캐스팅은 서로 다른 shape의 텐서끼리 연산할 수 있게 해준다.
* 불리언 마스킹은 조건에 맞는 데이터만 선택하거나 변경할 때 사용한다.
* `cat`은 기존 차원으로 이어 붙이고, `stack`은 새로운 차원을 만들어 쌓는다.
* `squeeze`는 크기가 1인 차원을 제거하고, `unsqueeze`는 차원을 추가한다.
* MLP는 Input Layer, Hidden Layer, Output Layer로 구성된다.
* Activation Function은 신경망에 비선형성을 추가한다.
* ReLU는 기울기 소실 문제를 완화하지만 Dying ReLU 문제가 있을 수 있다.
* Adam은 학습률을 동적으로 조정하는 자주 사용되는 옵티마이저이다.
