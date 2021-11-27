#pragma warning(disable:4996)

#include "RianDNN/RianDNN.h"
using namespace rian;

int main() {
	
	rian::HyperParm hyper_parm;
	hyper_parm.learning_rate_ = 0.1e-4f;
	/*
	  learning_rate_ = 0.1e-3f;
	  momentum_rate_ = 0.66f;
	  bias_init_ = 0.01;
	  loss_ = Loss::MSE;
	*/
	rian::Iterator iterator(&hyper_parm);

	rian::Layer input(1, Activation::None);
	rian::Layer h1(5, Activation::ReLU);
	rian::Layer h2(5, Activation::ReLU);
	rian::Layer h3(5, Activation::ReLU);
	rian::Layer output(1, Activation::None);
	iterator.add(&input, &h1);
	iterator.add(&h1, &h2);
	iterator.add(&h2, &h3);
	iterator.add(&h3, &output);
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
			iterator.run(sample, target);
			loss_mean += iterator.loss_;
			printf("%6.4lf  %6.4lf  %6.4lf  %10.8lf\n",
				sample[0], target[0], iterator.output_->result_[0], iterator.loss_);
		}

		//optimize_code
		for (int i = 0; i < iterator.output_->last_.size(); i++) {
			iterator.backprop(iterator.output_->last_[i], iterator.output_, 1);
		}

		//Monitoring
		printf("[[Monitoring]]\n");
		for (int i = 0; i < iterator.list_.size(); i++) {
			printf("(Layer %d)--------------------------\n", i);
			printf("bias : ");
			for (int j = 0; j < iterator.list_[i]->node_num_; j++) {
				printf("%10lf ", iterator.list_[i]->bias_[j]);
			}
			printf("\nresult : ");
			for (int j = 0; j < iterator.list_[i]->node_num_; j++) {
				printf("%10lf ", iterator.list_[i]->result_[j]);
			}
			printf("\ngrad : ");
			for (int j = 0; j < iterator.list_[i]->node_num_; j++) {
				printf("%10lf ", iterator.list_[i]->grad_[j] / iterator.execute_num_);
			}

			for (int j = 0; j < iterator.list_[i]->node_num_; j++) {
				if (iterator.list_[i]->next_.size() > 0) {
					printf("\n");
					for (int k = 0; k < iterator.list_[i]->next_[0]->node_num_; k++) {
						printf("%10lf ", iterator.list_[i]->connection_[0].weight_[j][k]);
					}
				}
			}
			printf("\n");
		}
		printf("loss_mean : %.10lf\n", loss_mean / iterator.execute_num_);
		
		iterator.grad_clear();
		//iterator.optimize();
		iterator.model_save();

		//pause
		if (epoch == 0) getchar();
		getchar();
	}

	printf("\nEnd of main\n");
	scanf("%*d");
	return 0;
}