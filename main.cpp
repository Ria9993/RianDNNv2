#pragma warning(disable:4996)

#include "FLSNN/FLSNN.h"
using namespace FLSNN;

int main() {
	

	/*
	{input, transition} += h1 += h2 += h3 += {output, transition}
	*/
	FLSNN::Layer input(128, "None");
	FLSNN::Layer transition(128, "ReLU");
	FLSNN::Layer h1(64, "ReLU");
	FLSNN::Layer h2(64, "ReLU");
	FLSNN::Layer h3(64, "ReLU");
	FLSNN::Layer output(2, "None");
	FLSNN::Iterator iterator;
	iterator.add(&input, &h1);
	iterator.add(&transition, &h1);
	iterator.add(&h1, &h2);
	iterator.add(&h2, &h3);
	iterator.add(&h3, &output);
	iterator.add(&h3, &transition);

	FLSNN::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.001f;
	hyper_parm.stochastic_rate_init_ = 0.3f;
	hyper_parm.backprop_depth_limit_ = 10;
	hyper_parm.grad_clipping_ = 100.0f;
	hyper_parm.momentum_rate_ = 0.5f;
	hyper_parm.loss_ = "MSE";
	hyper_parm.backprop_rate_ = 0.66f;
	hyper_parm.bias_init_ = 0.001f;

	iterator.build(&hyper_parm);

	vector<double> sample(128, 0.5);
	input.result_ = sample;
	
	iterator.run();

	return 0;
}