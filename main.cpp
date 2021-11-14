#pragma warning(disable:4996)

#include "RianDNN/RianDNN.h"
using namespace rian;

int main() {
	
	rian::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.001f;
	hyper_parm.stochastic_rate_init_ = 0.00f;
	hyper_parm.backprop_depth_limit_ = 10;
	hyper_parm.grad_clipping_ = 100.0f;
	hyper_parm.loss_ = "MSE";
	hyper_parm.backprop_rate_ = 0.8f;
	hyper_parm.bias_init_ = 0.01f;

	rian::Iterator iterator;
	iterator.hyper_parm_ = &hyper_parm;

	/*
	{input, transition} += h1 += h2 += h3 += {output, transition}
	*/
	rian::Layer input(1, "None");
	rian::Layer transition(64, "ReLU");
	rian::Layer h1(64, "ReLU");
	rian::Layer h2(64, "ReLU");
	rian::Layer h3(64, "ReLU");
	rian::Layer output(1, "None");
	iterator.add(&input, &h1);
	iterator.add(&transition, &h1);
	iterator.add(&h1, &h2);
	iterator.add(&h2, &h3);
	iterator.add(&h3, &output);
	iterator.add(&h3, &transition);
	iterator.output_ = &output;

	int cin;
	printf("1.new 2.load : ");
	scanf("%d", &cin);
	if (cin == 1) {
		iterator.init();
	}
	else if (cin == 2) {
		iterator.model_load();
	}

	vector<double> sample(128, 0.5);
	input.result_ = sample;
	vector<double> target(1, 0.3141592f);

	iterator.run(target);
	printf("%lf\n", output.result_[0]);

	iterator.model_save();

	//random_device rd;
	//mt19937 gen(rd());
	//uniform_real_distribution<double> rand(-1, 1);	


	scanf("%*d");
	return 0;
}