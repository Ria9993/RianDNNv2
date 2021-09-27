#pragma warning(disable:4996)

#include "FLSNN/FLSNN.h"
using namespace FLSNN;

int main() {
	FLSNN::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.001f;
	hyper_parm.stochastic_rate_init_ = 0.3f;
	
	FLSNN::Layer input(128, "None");
	FLSNN::Layer transition(128, "ReLU");
	FLSNN::Layer h1(64, "ReLU");
	FLSNN::Layer h2(64, "ReLU");
	FLSNN::Layer h3(64, "ReLU");
	FLSNN::Layer output(2, "None");

	/*
	{input, transition} += h1 += h2 += h3 += {output, transition}
	*/
	FLSNN::Iterator iterator;
	iterator.add(&input, &h1);
	iterator.add(&transition, &h1);
	iterator.add(&h1, &h2);
	iterator.add(&h2, &h3);
	iterator.add(&h3, &output);
	iterator.add(&h3, &transition);

	hyper_parm.backprop_depth_limit_ = 5;

	iterator.build(&hyper_parm);

	vector<double> sample(128, 0.5);
	input.result_ = sample;
	
	iterator.run();

	return 0;
}