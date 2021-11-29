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

rian::Model model(hyper_parm);

//input
model.add(Layer(1, Activation::None));
//hidden
model.add(Layer(5, Activation::ReLU));
model.add(Layer(5, Activation::ReLU));
//output
model.add(Layer(1, Activation::None));
```
### Run and Optimize
```cpp
void Model::run(vector<double>& input, vector<double>& target);
void Model::run(vector<double>& input);
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
### Predict Result
```cpp
vector <double> Model::predict();
```
### Model Save & Load
```cpp
//location : "\model.data"
void Model::model_save();
void Model::model_load();
```
