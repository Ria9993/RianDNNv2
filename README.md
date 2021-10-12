# Forward Looped Stochastic Neural Network
## Structure
![image](https://user-images.githubusercontent.com/44316628/135008221-ce130a1d-29ae-4116-a0f2-d4879c07987f.png)
![image](https://user-images.githubusercontent.com/44316628/135080997-7b7c71c9-6f0e-41bc-8918-c4627a8f7401.png)
<br>
실제 뉴런 시냅스의 활동전위는 단방향으로 동작하지만, 전체적으로 뇌는 순환적으로 동작하는데,<br>
그 이유는 인접한 뉴런끼리는 단방향으로 신호를 전달하지만, 신경망이 deep해지며 결국에는 원형적인 순환 구조를 띄게 됩니다.<br>
그리고 이 원형적인 순환 구조가 한 조직이라고 하였을때, 조직과 조직의 연결 또한 양방향적인 연결구조가 가능하게 됩니다.<br>
저자는 이를 인공신경망에 적용하기 위하여, 은닉(Hidden) 레이어 계층의 연산값이 다음 input으로 전달 됨으로서 원형순환적인 구조의 구현을 목표하고 있습니다.<br>
<br>
위에서 설명한 구조를 구현하기 위해 신경망의 다음 연산시 전이할 레이어를 "전이 레이어"(Transition Layer) 라고 정의하겠습니다.<br>
Transition 레이어는 이전 값을 저장함으로써 deep하지 않은 모델에서 deep한 연산을 해낼 수 있도록 하고<br>
이는 기억(memory) 으로서 저장 될 것으로 예상하고 있습니다.<br>
그리고 모든 은닉계층 레이어들에는 ReLU를 이용합니다. 이는 신경망에서 조직적인 연결이 조성되었을때, 조직끼리의 on-off를 유도합니다.<br>
이에 추가적으로, 실제 신경전달은 확률적이나 인공신경망은 그렇지 않기 때문에 stochastic_gate를 추가하여 확률적 신경전달 또한 사용합니다.<br>
<br>
### 문제점.<br>
1.계층이 순환됨에 따라, 역전파(backpropagation) 과정에 상당한 자원이 소모됩니다.<br>
2.stochastic_gate의 난수계산에도 상당한 자원이 소모됩니다.<br>
3.이러한 구조는 time-series 데이터셋만을 지향합니다.<br>
<br>
