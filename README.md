# 차례
## [1. 머신 러닝과 딥러닝](#1-머신-러닝과-딥러닝)
<details open>
  <summary>펼치기/접기</summary>

  [1-1. 머신 러닝과 딥러닝의 차이](#1-1-머신-러닝과-딥러닝의-차이)
</details>

## [2. 파이토치 기초](#2-파이토치-기초)
<details open>
  <summary>펼치기/접기</summary>


[2-1. 파이토치 특징 (09.04)](#2-1-파이토치-특징-0904)

[2-2. 파이토치 데이터셋 (09.04)](#2-2-파이토치-데이터셋-0904)

[2-3. 파이토치 모델 (09.04)](#2-3-파이토치-모델-0905)
</details>

## [3. 머신 러닝 알고리즘](#3-머신-러닝-알고리즘)
<details open>
  <summary>펼치기/접기</summary>

  [3-1. 지도학습 (09.05)](#3-1-지도학습-0905)

  * [3-1-1. k-최근접 이웃(knn) (09.05)](#3-1-1-k-최근접-이웃knn-0905)

  * [3-1-2. 서포트 벡터(SVM) (09.06)](#3-1-2-서포트-벡터-머신svm-0906)

  * [3-1-3. 결정 트리 (09.06)](#3-1-3-결정-트리-0906)

  * [3-1-4. 로지스틱 회귀 (09.06)](#3-1-4-로지스틱-회귀-0906)

  * [3-1-5. 선형 회귀 (09.06)](#3-1-5-선형-회귀-0906)

  [3-2. 비지도 학습 (09.06)](#3-2-비지도학습-0906)

  * [3-2-1. K-평균 군집화(KMC) (09.06)](#3-2-1-k-평균-군집화-0906)

  * [3-2-2. 주성분 분석(PCA) (09.06)](#3-2-2-주성분-분석pca-0906)
</details>

## #1. 머신 러닝과 딥러닝
### #1-1. 머신 러닝과 딥러닝의 차이


## #2. 파이토치 기초
### #2-1. 파이토치 특징 (09.04)
파이토치는 CPU/GPU 자원을 이용하여 텐서 조작 및 동적 신경망을 구축할 수 있는 프레임워크중 하나이다
파이토치는 Autograd, Aten, JIT 등의 C++ 엔진 등의 다양한 아키텍처로 이뤄져 있다.

파이토치에서는 기본적으로 텐서 연산 및 텐서 조작이 가능하다

------------------
### #2-2. 파이토치 데이터셋 (09.04)
파이토치를 사용하기에 앞서, 모델에 필요한 데이터셋을 불러올 때, 메모리에서 한번에 불러올 경우, 프로그램이 멈추거나 하는 등 효율적이지 않기 때문에 `데이터셋`을 만들어 사용한다.

또한 파이토치에서 제공하는 MNIST 등을 사용할 수도 있다
``` python
# 데이터셋 예시
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.label = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx, 0:3]).int()
        label = torch.tensor(self.label.iloc[idx, 3]).int()
        return sample, label
tensor_dataset = MyDataset('./test.csv')
dataset = DataLoader(tensor_dataset, batch_size=4, shuffle=True)
```

------------
### #2-3. 파이토치 모델 (09.05)
모델은 다음과 같은 요소들로 이뤄져 있다.
* 계층(layer):
    * 가중치(weight) 와 편차(bias) 를 가져 연산을 수행한다
    * 특정 개수의 입력 노드로부터 연산을 거쳐, 또 다른 개수의 출력노드로 값이 도출된다
    * 합성곱층, 선형계층 등이 있다
* 모듈(module):
    * 계층이 모여 구성된 것으로, 모듈이 모여 모듈을 구성할 수 있다
* 모델(model):
    * 최종적으로 원하는 네트워크로, 한 개의 모듈이 모델 그 자체가 될 수도 있다

모델을 구현할 때, 대부분 Module 을 상속받아 사용하는데 그 경우 `__init__` 에서 모듈, 활성화 함수 등을 정의하고, `forawrd` 에서는 모델에서 실행될 연산을 정의한다
```python
# single layer, single module
class SLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layer = nn.Linear(
            in_features=inputs, 
            out_features=1
        )
        self.activation = nn.Sigmoid
    
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

# multi layer, single module
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=30,
                kernel_size=5
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(
                in_features=30 * 5 * 5,
                out_features=10
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x
```

## #3. 머신 러닝 알고리즘
### #3-1. 지도학습 (09.05)
지도학습은 모델을 훈련할 때, 사전에 입력되는 데이터에 정답을 알려 주고 학습을 하는 방법이다

지도학습의 종류
- 분류
    - 이산형 데이터를 받아 사전에 훈련받은 데이터들의 레이블 중 하나로 예측하는 방식이다
- 회귀
    - 연속된 데이터를 받아 연속된 값을 예측하여 연속된 값을 예측하는 방식으로, 보통 흐름에 따라 연속적으로 변하는 값을 예측할 때 사용한다

### 3-1-1. k-최근접 이웃(KNN) (09.05)
k 최근접 이웃은 미리 라벨이 붙은 클러스터 들 중에, 새로운 입력 데이터가 있으면 해당 데이터 근처 `k`개의 데이터를 보고, 새로운 데이터에 어떤 라벨이 붙일지를 정하는 알고리즘 이다

k값에 따라 비교할 데이터 대상이 달라져 결과가 크게 달라짐으로 초기 설정이 중요하다


``` python
# 모델을 생성 및 훈련하고, 정확도를 계산하는 예시
classifier = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
```

-----
### #3-1-2. 서포트 벡터 머신(SVM) (09.06)
서포트 벡터 머신이란, 데이터 클러스터에서 분류를 위한 기준점인 `결정 경계`를 결정하는 알고리즘이다

결정 경계로부터 가장 가까이 있는 데이터를 `서포트 벡터` 라고 하고, 서포트 벡터와 결정 경계 사이의 거리를 `마진` 이라고 하는데

최적의 결정 경계를 정하기 위해서는 마진이 최대가 되도록 해야 한다
``` python
# SVM 모델 생성 및 훈련 예시
svm = svm.SVC(kernel='linear', C=1, gamma=0.5)
# 훈련 데이터로 svm 모델 훈련
svm.fit(x_train, y_train)
# 테스트데이터로 예측
predictions = svm.predict(x_test)
score = metrics.accuracy_score(y_test, predictions)
```
예시에서 보면 `c` 와 `gamma` 가 있는데

`C` 값은 오류를 얼마나 허용할지를 정하는 파라미터이며, 클 수록 하드마진이다

`gamma` 값은 각 결정 경계를 얼마나 유연하게 가져갈지, 즉 경계의 곡선이 얼마나 휘어질지를 정하는 파라미터로 값이 클수록 급격하게 휜다.
하지만 해당 값이 너무 클 경우, 훈련 데이터에 많이 의존하기 때문에 과적합을 초래할 수 있으니 주의해야 한다


### ※ 추가 정보
비선형 문제, 즉 결정 곡선이 비선형일 때 찾는 방법이 저차원 데이터를 고차원으로 보내는 것인데, 이것은 연산량이 너무 많아 다음과 같은 커널 트릭으로 해결한다

>선형 커널(linear kernel): 
>   > 선형으로 분류 가능한 데이터에 적용하며 커널 트릭을 사용하지 않겠다는 의미이다.
>   >
>   > $$K(a, b) = a^T * b$$
>   >
>   > $(a, b)$는 입력 벡터

> 다항식 커널(polynomial kernel):
>   > 실제로는 특정을 추가하지 않지만, 다항식 특성을 많이 추가한 것과 같은 결과를 얻을 수 있는 방법이다. 때문에 고차원 매핑이 가능하다
>   >
>   > $$K(a, b) = (\gamma a^t * b)^d$$
>   >
>   > $\begin{pmatrix} a, b & 입력 벡터 \\ \gamma & 감마 \\ d  & 차원 \end{pmatrix}$ 단, 이때 $\gamma, d$는 하이퍼파라미터

> 가우시안 RBF 커널(Gaussian RBF kernel):
>   > 입력 벡터를 차원이 무한한 고차원으로 매핑하는 것으로 모든 차수의 모든 다항식을 고려, 다항식 커널은 차수에 한계가 있는 문제를 해결
>   > 
>   > $$K(a, b) = \exp(-\gamma \rVert a= b\rVert ^ 2)$$
>   >
>   > 이때 $\gamma$ 는 하이퍼파라미터

--------
### #3-1-3. 결정 트리 (09.06)
결정 트리는 데이터를 분류하거나 결과를 예측하는 분석 방법이다

_결정 트리 예시_
| 자동차 | 자전거 | 참새 | 사람 |
| - | - | - | - |
| 엔진이 있다 | 엔진이 없다 | 날개가 있다 | 날개가 없다 |
| 바퀴가 있다  | ` | 바퀴가 없다 | ` |

결정 트리는 데이터를 1차로 분류한 후 각 영역의 순도가 증가하고, 불순도와 불확실성은 감소하는 방향으로 학습을 진행시킨다.

이중, 순도는 범주 안에 같은 데이터가 모여있는 정도이고 불순도는 계산을 통해 구한다
``` python
# 결정 트리 예시
# 결정 트리 모델 생성
model = tree.DecisionTreeClassifier()

# 모델 훈련
model.fit(x_train, y_train)

# 모델 예측
y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict))

# 혼동 행렬로 성능측정
print(pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Pred Negative', 'Pred Positive'],
    index=['Actual Negative', 'Actual Positive']
))
```
※ 혼동 행렬이란 True/False, Positive/Negative 의 조건으로
* 예측값이 Positive 인데 실제값도 Positive 인 경우
* 예측값이 Positive 인데 실제값은 Negative 인 경우
* 예측값이 Negative 인데 실제값은 Positive 인 경우
* 예측값이 Negative 인데 실제값도 Negative 인 경우

를 표현하는 행렬이다

----
### #3-1-4. 로지스틱 회귀 (09.06)
회귀란 두 변수에서 한 변수로 다른 변수를 예측하거나 두 변수의 관계를 규명할 때 사용하는 방법으로 이 때 사용하는 변수는 다음과 같다
* **독립 변수(예측 변수)**: 영향을 미칠 것으로 예상되는 변수
* **종속 변수(기준 변수)**: 영향을 받을 것으로 예상되는 변수

예시로는 몸무게(종속 변수) 와 키(독립 변수)가 있다

로지스틱 회귀는 일반적인 회귀와는 다르게 **분석하고자 하는 대상들이 두 집단 혹은 그 이상의 집단으로 나누어진 경우, 개별 관측치들이 어느 집단으로 분류될 수 있는지 분석하고 이를 예측하는 모형을 개발**하는데 사용되는 통계 기법이다.
``` python
#로지스틱 회귀 모델

# 로지스틱 회귀 모델 생성
logisticRegr = LogisticRegression()

# 훈련
logisticRegr.fit(x_train, y_train)

# 테스트셋을 사용해 모델 예측
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print('score: ', score)
# 추가적으로 혼동행렬을 이용해 시각화 할 수도 있다
```

---
### #3-1-5. 선형 회귀 (09.06)
선형 회귀는 독립 변수와 종속 변수가 선형 관계를 가질 때 사용하면 유용하며, 선형 특징상 복잡한 과정이 없어 제한된 환경에서도 사용할 수 있다

로지스틱 회귀와의 차이는 선형 회귀는 변수 x 와 y 의 관계가 직선으로 나타나며 때문에 예측값 y는 0~1 을 초과할 수 있다

하지만 로지스틱 회귀는 x 와 y 의 관계가 S-커브 로 나타나며 예측값은 0~1 사이이다 (종속변수가 예/아니오 로 나타나기 때문)
``` python
# 선형 회귀 모델

# 선형 회귀 모델 생성
regressor = LinearRegression()

# 훈련
regressor.fit(x_train, y_train)

# 모델 예측
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actural': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# 테스트셋으로 회귀선 표현
plt.scatter(x_test, y_test, color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()
```

선형 회귀 모델을 평가할 때는 평균 제곱법과 루트평균 제곱법을 사용하여 모델을 평가한다

평균 제곱법이 $\mathrm{MSE} = \frac{1}{n}\sum_{l=1}^N(y_i- \check{y_i})^2$ 라면,

루트 평균 제곱법은 $\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{l=1}^N(y_i- \check{y_i})^2}$ 로 전체에 루트만 씌운것과 같다

---
### #3-2. 비지도학습 (09.06)
비지도 학습은 분류되거나 레이블을 붙이지 않은 데이터로 훈련시키는 학습이며 

비지도 학습에는 군집(cluster)과 축소(dimensionality reduction) 이 있다

군집은 데이터를 그룹화 하여 분류하는데 사용하고,

차원 축소는 데이터를 압축하거나, 필요한 속성을 도출해내는데 사용한다

---
### #3-2-1. K 평균 군집화 (09.06)
K 평균 군집화는 데이터를 입력받아 여러 그룹으로 묶는 알고리즘 이다.

해당 알고리즘은 데이터를 받아 각 데이터에 레이블을 할당해 클러스터링을 수행하는데 학급 과정은 다음과 같다.
1. **중심점 선택**: 랜덤하게 초기 중심점을 K개 선택한다
2. **클러스터 할당**: K개의 중심점과 각각 데이터간의 거리를 측정 후, 가장 가까운 중심점을 기준으로 데이터를 할당 하는것으로 클러스터화 하여 레이블을 할당한다
3. **새로운 중심점 선택**: 클러스터마다 새로운 중심점을 계산한다.
4. **범위 확인**: 선택된 중심점에 변화가 없다면 진행을 멈추고, 있다면 2~3 과정을 반복한다

하지만 K-평균 군집화 알고리즘은 다음 상황에서는 사용하지 않는것이 권장된다
* **데이터가 비선형일때**: 해당 알고리즘은 각 클러스터간의 거리가 가장 중요하게 동작하는데, 거리라는 조건에 따라 클러스터를 설정하는 행위는 선형적이라고까지 할 수 있게 동작하기 때문에, 데이터가 비 선형적이라면 클러스터가 정상적으로 형성되지 않을 가능성이 높다
* **군집 크기가 다를때**: 군집 크기가 다르다면 자연스레 큰 군집의 외각에 있는 데이터가 해당 클러스터의 중심점과 거리가 멀어 다른 클러스터로 합쳐질 가능성이 높은데, 해당 경우가 많이 발생하면 클러스터가 원하는 대로 형성되지 않게 된다
* **군집마다 밀집도와 거리가 다를 때**: 위와 거의 동일하다. 밀집도가 낮은 클러스터의 외각에 있는 데이터는 해당 클러스터의 중심점과 거리가 멀어 다른 클러스터와 합쳐지며 데이터가 오염된다
``` python
#KMC
km = KMeans(n_clusters=k)
km = km.fit(data_transformed)
print('거리 제곱의 합:', km.inertia_)
```
거리 제곱의 합(Sum of Squared Distances) 은 가장 가까운 클러스터 중심까지 거리를 제곱한 값을 구할 때 사용하며 다음과 같은 수식이다
$$\mathrm{SSD} = \sum_{x, y} (I_1(x, y) - I_2(x, y))^2$$
K 값이 증가하면 당연히 클러스터의 개수가 많아지며 SSD는 0에 가까워지는 경향이 있다

**※ 추가 정보**

KMC 의 단점으로 소수의 데이터가 적절한 클러스터와 거리가 멀리 떨어져 있는, 즉 오목하거나 볼록한 부분을 잘 처리하지 못한다는 점이 있는데
연산량은 조금 더 많지만 이런 노이즈와 이상치를 잘 처리할 수 있는 `밀도 기반 군집 분석 (DBSCAN)` 이 있다

---
### #3-2-2. 주성분 분석(PCA) (09.06)
PCA 는 고차원 데이터에서는 중요하지 않은 변수가 많아지고 성능도 나빠지는 경향이 있어 고차원 데이터를 저차원으로 축소시켜 데이터의 대표 특성만 추출하는 알고리즘 이다.

차원 축소는 다음과 같은 단계로 진행된다
1. **데이터들의 분포 특성을 잘 설명하는 벡터 2개 선택**:
간단하게는 원형으로 된 클러스터가 있다면 원의 중심을 수직으로 지나는 벡터 2개를 예시로 들 수 있는데, 해당 벡터들의 방향과 크기로 클러스터의 위치, 모양을 예상할 수 있기 때문이다
2. **벡터 2개를 위한 가중치를 찾을 때까지 학습**:
즉 PCA는 데이터 하나하나의 성분이 아닌, 여러 데이터가 모인 클러스터에서 해당 클러스터의 주성분을 분석하는 방법이기 때문이다
``` python
# PCA 학습 예시
# 2차원으로 차원 축소 선언
pca = PCA(n_components=2)
x_principal = pca.fit_transform(x_normalized)
x_principal = pd.DataFrame(x_principal)
x_principal.columns = ['P1', 'P2']

# 모델 튜닝
db = DBSCAN(eps=0.0375, min_samples=50).fit(x_principal)
# min_samples 수를 변경해서 큰 값을 넣는다면 작은 규모의 클러스터가 무시된다

labels = db.labels_

colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
cvec = [colors[l] for l in labels]

plt_color = [
    plt.scatter(x_principal['P1'], x_principal['P2'],marker='o', color=c)
        for c in colors1
]
plt.figure(figsize=(9, 9))
plt.scatter(x_principal['P1'], x_principal['P2'], c=cvec)
plt.legend(
    plt_color,
    ['Label' + str(i) for i in range(7)],
    scatterpoints=1,
    loc='upper left',
    ncol=3,
    fontsize=8
)
plt.show()
```