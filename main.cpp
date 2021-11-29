#pragma warning(disable:4996)

#include "RianDNN/RianDNN.h"
using namespace rian;

int main() {
	
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

	int ci;
	printf("1.new 2.load : ");
	scanf("%d", &ci);
	switch (ci) {
	case 1 :
			model.init();
			break;
	case 2 :
			model.model_load();
			break;
	}

	//{y = 2x} Example
	for (int epoch = 0; epoch < 1000; epoch++) {

		//mini-batch
		double loss_mean = 0;
		printf("input  |target  |predict |loss\n");
		for (int i = 0; i < 100; i++) {
			//random
			random_device rd;
			mt19937 gen(rd());
			uniform_real_distribution<double> rnd(0, 1);

			vector<double> sample(1, rnd(gen));
			vector<double> target(1, 2 * sample[0]);

			//run
			model.run(sample, target);
			loss_mean += model.loss_;
			printf("%6.4lf  %6.4lf  %6.4lf  %10.8lf\n",
				sample[0], target[0], model.layer_.rbegin()->result_[0], model.loss_);
		}

		//Monitoring
		printf("[[Monitoring]]\n");
		for (int i = 0; i < model.layer_.size(); i++) {
			printf("(Layer %d)--------------------------\n", i);
			printf("bias : ");
			for (int j = 0; j < model.layer_[i].node_num_; j++) {
				printf("%10lf ", model.layer_[i].bias_[j]);
			}
			printf("\nresult : ");
			for (int j = 0; j < model.layer_[i].node_num_; j++) {
				printf("%10lf ", model.layer_[i].result_[j]);
			}
			printf("\ngrad : ");
			for (int j = 0; j < model.layer_[i].node_num_; j++) {
				printf("%10lf ", model.layer_[i].grad_[j] / model.execute_num_);
			}
			printf("\n");
		}
		printf("learning_rate : %.10lf\n", hyper_parm.learning_rate_);
		printf("loss_mean : %.10lf\n", loss_mean / model.execute_num_);
		
		//model.grad_clear();
		model.optimize();
		model.model_save();

		//pause
		if (epoch == 0) getchar();
		getchar();
	}

	printf("\nEnd of main\n");
	scanf("%*d");
	return 0;
}