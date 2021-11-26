#pragma warning(disable:4996)

#include "RianDNN/RianDNN.h"
using namespace rian;

int main() {
	
	rian::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.001f;
	/* //Default hyper_parm
	learning_rate_ = 0.001f;
	grad_clipping_ = 100.0f;
	backprop_depth_limit_ = 100;
	momentum_rate_ = 0.8f;
	loss_ = "MSE";
	backprop_rate_ = 0.66f; ///< develop
	stochastic_rate_init_ = -1.0f;
	bias_init_ = 0.01f;
	*/

	rian::Iterator iterator(&hyper_parm);

	rian::Layer input(1, Activation::None);
	rian::Layer h1(3, Activation::ReLU);
	rian::Layer h2(3, Activation::ReLU);
	rian::Layer output(1, Activation::None);
	iterator.add(&input, &h1);
	iterator.add(&h1, &h2);
	iterator.add(&h2, &output);
	iterator.output_ = &output;

	int ci;
	printf("1.new 2.load : ");
	scanf("%d", &ci);
	switch (ci) {
	case 1 :
			iterator.init();
			break;
	case 2 :
			iterator.model_load();
			break;
	}

	//{y = 2x} Example
	for (int epoch = 0; epoch < 1000; epoch++) {

		//mini-batch
		printf("input  |target  |predict |loss\n");
		for (int i = 0; i < 10; i++) {
			//random
			random_device rd;
			mt19937 gen(rd());
			uniform_real_distribution<double> rnd(0, 1);

			vector<double> sample(1, rnd(gen));
			vector<double> target(1, 2 * sample[0]);

			//run
			iterator.run(sample, target);
			printf("%6.4lf  %6.4lf  %6.4lf  %6.4lf\n",
				sample[0], target[0], iterator.output_->result_[0], iterator.loss_);
		}
		iterator.optimize();
		iterator.model_save();

		printf("bias : %lf\n weight : %lf %lf %lf\n",
			iterator.list_[0]->bias_[0],
			iterator.list_[0]->connection_[0].weight_[0][0],
			iterator.list_[0]->connection_[0].weight_[0][1], 
			iterator.list_[0]->connection_[0].weight_[0][2]);
		
		//pause
		if (epoch == 0) getchar();
		getchar();
	}

	printf("\nEnd of main\n");
	scanf("%*d");
	return 0;
}