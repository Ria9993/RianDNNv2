# Stochastic Bidirectional Recurrent Neural Network
## Structure
This model is designed differently from the existing RNN models. <br>
It's similar to the existing DNN models with FC layers, but other concepts are as follows.
#### 1. Only adjacent neurons can transmit signals.
#### 2. Signal transmission between neurons is possible in bidirections for circulation.
#### 3. Add a stochastic transmission gate that mimics a stochastic action potential.
#### 4. To induce a cycle, the input layer is only connected to half of the next layer.
![image](https://user-images.githubusercontent.com/44316628/133630038-2b1b76cf-38cd-4691-b40d-b2b119dc3425.png)
![image](https://user-images.githubusercontent.com/44316628/133631075-495e36be-9678-4127-84cc-4ea294356097.png)
