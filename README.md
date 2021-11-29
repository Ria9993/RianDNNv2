# RianDNNv2
#### Header-only DNN library for C++ <br/>

# Features

### Parallel Compute
- Parallel compute is supported by default.(C++ AMP Concurrency) <br/>
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
hyper_parm.learning_rate_ = 0.1e-3f;
/*
	learning_rate_ = 0.1e-3f;
	learning_rate_schedule_ = 0.97;
	momentum_rate_ = 0.66f;
	bias_init_ = 0.01;
	loss_ = Loss::MSE;
*/

rian::Model iterator(&hyper_parm);

rian::Layer input(1, Activation::None);
rian::Layer h1(5, Activation::ReLU);
rian::Layer h2(5, Activation::ReLU);
rian::Layer h3(5, Activation::ReLU);
rian::Layer output(1, Activation::None);
model.add(&input, &h1);
model.add(&h1, &h2);
model.add(&h2, &h3);
model.add(&h3, &output);
model.output_ = &output;
```
### Run and Optimize
```cpp
void Model::run(vector<double>& input, vector<double>& target);
void Model::optimize();
```
```cpp
/*example y = 2x */
//set input & target
vector<double> sample(1, rand_real());
vector<double> target(1, 2 * sample[0]);

//run & optimize
model.run(sample, target);
model.optimize();
```
### Evaluating
```cpp
vector <double> Model::predict();
```
### Model Save & Load
```cpp
//location : "\model.data"
void model_save();
void model_load();
```
