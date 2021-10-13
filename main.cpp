#pragma warning(disable:4996)

#include "FLSNN/FLSNN.h"
using namespace FLSNN;

int main() {
	

	/*
	{input, transition} += h1 += h2 += h3 += {output, transition}
	*/
	FLSNN::Layer input(1, "None");
	FLSNN::Layer transition(64, "ReLU");
	FLSNN::Layer h1(64, "ReLU");
	FLSNN::Layer h2(64, "ReLU");
	FLSNN::Layer h3(64, "ReLU");
	FLSNN::Layer output(1, "None");
	FLSNN::Iterator iterator;
	iterator.add(&input, &h1);
	iterator.add(&transition, &h1);
	iterator.add(&h1, &h2);
	iterator.add(&h2, &h3);
	iterator.add(&h3, &output);
	iterator.add(&h3, &transition);
	iterator.output_ = &output;

	FLSNN::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.001f;
	hyper_parm.stochastic_rate_init_ = 0.00f;
	hyper_parm.backprop_depth_limit_ = 10;
	hyper_parm.grad_clipping_ = 100.0f;
	hyper_parm.momentum_rate_ = 0.5f;
	hyper_parm.loss_ = "MSE";
	hyper_parm.backprop_rate_ = 0.8f;
	hyper_parm.bias_init_ = 0.01f;
	iterator.hyper_parm_ = &hyper_parm;

	iterator.build();

	vector<double> sample(128, 0.5);
	input.result_ = sample;
	vector<double> target(1, 0.3141592f);

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> rand(-1, 1);
	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < 10; j++) {
			vector<double>sample(1, rand(gen));
			if (j % 10 == 0)
				sample[0] = 0.5f;
			input.result_ = sample;
			vector<double>target(1, sample[0] * sample[0]);
			iterator.run(target);
			printf("[%d] input : %lf, output : %lf, answer : %lf loss : %lf\n", i*10+j, sample[0], output.result_[0], target[0], iterator.loss_);
		}
		iterator.optimize();
	}


	scanf("%*d");
	return 0;
}