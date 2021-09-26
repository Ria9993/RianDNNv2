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
	input += h1 += h2 += h3 += output;
	h3 += transition += h1;

	hyper_parm.backprop_depth_limit_ = 5;

	input.build(&hyper_parm);

	return 0;
}