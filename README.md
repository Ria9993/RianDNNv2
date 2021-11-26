# RianDNNv2
#### Header-only DNN library for C++ <br/>
I'm making this for use in my machine learning projects <br/>

# Features

### Optimizer
```mk
Momentum
```
### Activation Function
```mk
ReLU, 
None
//TODO : Sigmoid, Softmax
```
### Loss Function
```mk
MSE(Mean Squared)
//TODO : CEE(Cross Entropy)
```
### Weight Initialize
```mk
He Normal(Default)
```
### Layer
```mk
Dense(Default)
//TODO :
```

# Example

### Include

```cpp
#include "RianDNN/RianDNN.h"
using namespace rian;
```

### Create Model

```cpp
rian::HyperParm hyper_parm;
hyper_parm.learning_rate_ = 0.1e-4f;
/*
  learning_rate_ = 0.1e-3f;
  momentum_rate_ = 0.66f;
  bias_init_ = 0.01;
*/

rian::Iterator iterator(&hyper_parm);

rian::Layer input(1, Activation::None);
rian::Layer h1(5, Activation::ReLU);
rian::Layer h2(5, Activation::ReLU);
rian::Layer h3(5, Activation::ReLU);
rian::Layer output(1, Activation::None);
iterator.add(&input, &h1);
iterator.add(&h1, &h2);
iterator.add(&h2, &h3);
iterator.add(&h3, &output);
iterator.output_ = &output;
```
### Run and Optimize
```cpp
void Iterator::run(vector<double>& input, vector<double>& target);
void Iterator::optimize();
```
```cpp
/*example y = 2x */
//set input & target
vector<double> sample(1, rand_real());
vector<double> target(1, 2 * sample[0]);

//run & optimize
iterator.run(sample, target);
iterator.optimize();
```
### Evaluating
```cpp
vector <double> Iterator::predict();
```
### Model Save & Load
```cpp
//location : "\model.data"
void model_save();
void model_load();
```
