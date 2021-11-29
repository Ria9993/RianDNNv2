#pragma warning(disable:4996)

#include "RianDNN/RianDNN.h"
using namespace rian;

int main() {
	
	rian::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.1e-3f;
	/*
	  learning_rate_ = 0.1e-3f;
	  learning_rate_schedule_ = 0.97;
	  momentum_rate_ = 0.8f;
	  bias_init_ = 0.01;
	  loss_ = Loss::MSE;
	*/
	rian::Model model(&hyper_parm);

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
				sample[0], target[0], model.output_->result_[0], model.loss_);
		}

		//Monitoring
		printf("[[Monitoring]]\n");
		for (int i = 0; i < model.list_.size(); i++) {
			printf("(Layer %d)--------------------------\n", i);
			printf("bias : ");
			for (int j = 0; j < model.list_[i]->node_num_; j++) {
				printf("%10lf ", model.list_[i]->bias_[j]);
			}
			printf("\nresult : ");
			for (int j = 0; j < model.list_[i]->node_num_; j++) {
				printf("%10lf ", model.list_[i]->result_[j]);
			}
			printf("\ngrad : ");
			for (int j = 0; j < model.list_[i]->node_num_; j++) {
				printf("%10lf ", model.list_[i]->grad_[j] / model.execute_num_);
			}

			for (int j = 0; j < model.list_[i]->node_num_; j++) {
				if (model.list_[i]->next_.size() > 0) {
					printf("\n");
					for (int k = 0; k < model.list_[i]->next_[0]->node_num_; k++) {
						printf("%10lf ", model.list_[i]->connection_[0].weight_[j][k]);
					}
				}
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