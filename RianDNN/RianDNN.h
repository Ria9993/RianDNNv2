#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
using namespace std;

#include <thread>
#include <ppl.h>
using namespace concurrency;

namespace rian {

	enum class Loss {
		MSE, ///< Mean Squared Error
		CEE ///< Cross Entropy Error
	};

	class HyperParm {
	private:
	public:
		double learning_rate_;
		double learning_rate_schedule_; ///< Update learning_rate_ every time to optimize
		double momentum_rate_;
		Loss loss_;
		double bias_init_; ///< bias init value

		HyperParm() {
			//Default optional parametor set
			learning_rate_ = 0.1e-3f;
			learning_rate_schedule_ = 0.97;
			momentum_rate_ = 0.66f;
			loss_ = Loss::MSE;
			bias_init_ = 0.01;
		}
	};

	class Connection {
	private:
	public:

		//Element
		vector<vector<double>> weight_;

		//Backprop Gradient
		vector<vector<double>> weight_grad_;
		vector<vector<double>> weight_grad_momentum_;
	};

	enum class Activation {
		None,
		ReLU,
		Sigmoid
	};

	class Layer {
	private:
	public:
		//Parametor
		int node_num_;
		Activation activation_;

		//Connection
		Connection connection_;

		//Element
		int execute_num_;
		vector<double> bias_;
		vector<double> calc_result_;
		vector<double> result_;

		//Backprop Gradient
		vector<double> grad_;
		vector<double> grad_momentum_;


		//Layer init
		Layer(int node_num, Activation activation) {
			node_num_ = node_num;
			activation_ = activation;
		}

		Layer() {

		}

	};

	class Model {
	private:
	public:
		//element
		vector<Layer> layer_; ///< 중복 없는 layer_list
		HyperParm hyper_parm_; ///< pointer of hyper_parm
		int execute_num_; ///< run 횟수
		double loss_;

		Model() {
			execute_num_ = 0;
			loss_ = 0;
		}
		Model(HyperParm hyper_parm) {
			hyper_parm_ = hyper_parm;
			execute_num_ = 0;
			loss_ = 0;
		}

		//Function
		void add(Layer layer);
		void init(bool load_flag = false);
		void run(vector<double>& input, vector<double>& target);
		void run(vector<double>& input);
		void calc(bool grad_calc_flag);
		void optimize();
		void backprop();
		void grad_clear();
		void model_save(string filename); ///< file save & load
		void model_load(string filename);
		vector <double> predict();
		void grad_copy(Model& source);

	};

	void Model::add(Layer layer) {

		layer_.push_back(layer);

		return;
	}

	void Model::init(bool load_flag) {

		for (int layer_i = 0; layer_i < layer_.size(); layer_i++) {

			Layer* layer = &layer_[layer_i];

			//element init
			layer->execute_num_ = 0;
			layer->bias_.resize(layer->node_num_, hyper_parm_.bias_init_);
			layer->calc_result_.resize(layer->node_num_, 0);
			layer->result_.resize(layer->node_num_, 0);
			layer->grad_.resize(layer->node_num_, 0);
			layer->grad_momentum_.resize(layer->node_num_, 0);

			//connection init
			random_device rd;
			mt19937 gen(rd());

			//output_layer outofrange
			if (layer_i == layer_.size() - 1) {
				break;
			}


			layer->connection_.weight_.resize(layer->node_num_, vector<double>(layer_[layer_i + 1].node_num_));
			layer->connection_.weight_grad_.resize(layer->node_num_, vector<double>(layer_[layer_i + 1].node_num_, 0));
			layer->connection_.weight_grad_momentum_.resize(layer->node_num_, vector<double>(layer_[layer_i + 1].node_num_, 0));

			//Weight, stochastic_gate init
			if (load_flag == false) { ///< 모델 불러오기 시 초기화 배제

				normal_distribution<double> HE(0, sqrtf((double)2 / layer->node_num_)); ///< HE initialization
				for (int j = 0; j < layer->node_num_; j++) {
					for (int k = 0; k < layer_[layer_i + 1].node_num_; k++) {

						layer->connection_.weight_[j][k] = HE(gen);
					}
				}
			}
		}

		return;
	}

	void Model::run(vector<double>& input) {

		//input set
		layer_[0].result_ = input;

		//reset calc_result
		for (int i = 0; i < layer_.size(); i++) {
			layer_[i].calc_result_ = layer_[i].bias_;
		}

		calc(false);

		return;
	}

	void Model::run(vector<double>& input, vector<double>& target) {

		//input set
		layer_[0].result_ = input;

		//reset calc_result
		for (int i = 0; i < layer_.size(); i++) {
			layer_[i].calc_result_ = layer_[i].bias_;
		}

		//calc all layers
		calc(true);

		//calc loss and derivative
		loss_ = 0;
		switch (hyper_parm_.loss_) {
		case Loss::MSE:
			for (int i = 0; i < layer_.rbegin()->node_num_; i++) {

				double error = layer_.rbegin()->result_[i] - target[i];
				loss_ += error * error;

				//derivative
				layer_.rbegin()->grad_[i] += (2 / layer_.rbegin()->node_num_) * error;
			}
			loss_ /= layer_.rbegin()->node_num_;
			break;
		case Loss::CEE:
			for (int i = 0; i < layer_.rbegin()->node_num_; i++) {

				double tmp = target[i] * log2f(layer_.rbegin()->result_[i]);
				if (!isnan(tmp))
					loss_ -= tmp;

				//derivative
				layer_.rbegin()->grad_[i] += layer_.rbegin()->result_[i] - target[i];
			}
			loss_ /= layer_.rbegin()->node_num_;
			break;
		default:
			break;
		}

		execute_num_++;
	}

	void Model::calc(bool grad_calc_flag) {
		//random_device rd;
		//mt19937 gen(rd());

		for (int layer_i = 0; layer_i < layer_.size(); layer_i++) {

			Layer* source = &layer_[layer_i];

			//Activation
			for (int i = 0; i < source->node_num_; i++) {
				switch (source->activation_) {
				case Activation::ReLU:
					source->result_[i] = fmax(0, source->result_[i]);
					//source->grad_[i] += derivative(Activation::ReLU, source->result_[i]);
					break;
					//case Activation::LeakyReLU :
					//	source->result_[i] = fmax(source->result_[i] * 0.01, source->result_[i]);
					//	break;
				case Activation::Sigmoid:
					break;
				case Activation::None:
					//source->grad_[i] += 1;
					break;
				default:
					break;
				}
			}

			//Not calc output_layer
			if (layer_i == layer_.size() - 1) {
				break;
			}

			Layer* dest = &layer_[layer_i + 1];

			//multi-threaded Calculate
			if (grad_calc_flag) {
				parallel_for(0, dest->node_num_, [&](int n) {
					for (int j = 0; j < source->node_num_; j++) {

						//Weight
						dest->calc_result_[n] += source->result_[j] * source->connection_.weight_[j][n];
						//Gradient
						source->connection_.weight_grad_[j][n] += source->result_[j];
					}
					});
			}
			else {
				parallel_for(0, dest->node_num_, [&](int n) {
					for (int j = 0; j < source->node_num_; j++) {

						//Weight
						dest->calc_result_[n] += source->result_[j] * source->connection_.weight_[j][n];
					}
					});
			}


			//copy calc_result to result
			dest->result_ = dest->calc_result_;

		}

		return;
	}

	void Model::optimize()
	{

		//backprop_chain
		backprop();
		grad_clear();

		//update learning_rate schedule
		hyper_parm_.learning_rate_ *= hyper_parm_.learning_rate_schedule_;

		return;
	}

	void Model::backprop()
	{

		for (int layer_i = layer_.size() - 1 - 1; layer_i >= 0; layer_i--) {

			Layer* layer = &layer_[layer_i];
			Layer* source = &layer_[layer_i + 1];

			//calc grad & elements update
			for (int i = 0; i < layer->node_num_; i++) {
				for (int j = 0; j < source->node_num_; j++) {

					double grad_tmp;
					//weight
					layer->connection_.weight_grad_momentum_[i][j] *= hyper_parm_.momentum_rate_;
					grad_tmp = (source->grad_[j] * layer->connection_.weight_grad_[i][j]) / execute_num_;
					layer->connection_.weight_grad_momentum_[i][j] += grad_tmp;
					layer->connection_.weight_[i][j] -= hyper_parm_.learning_rate_ * layer->connection_.weight_grad_momentum_[i][j];

					//backprop
					layer->grad_[i] += grad_tmp;
				}
			}

			//bias update
			for (int i = 0; i < layer->node_num_; i++) {
				layer->grad_momentum_[i] *= hyper_parm_.momentum_rate_;
				layer->grad_momentum_[i] += layer->grad_[i] / execute_num_;
				layer->bias_[i] -= hyper_parm_.learning_rate_ * layer->grad_momentum_[i];
			}

		}

		return;
	}

	void Model::grad_clear()
	{
		for (int i = 0; i < layer_.size(); i++) {

			for (int j = 0; j < layer_[i].node_num_; j++) {

				layer_[i].grad_[j] = 0;

				//output layer outofrange
				if (i == layer_.size() - 1) {
					break;
				}
				for (int l = 0; l < layer_[i + 1].node_num_; l++) {

					layer_[i].connection_.weight_grad_[j][l] = 0;
				}
			}
		}

		execute_num_ = 0;

		return;
	}

	void Model::grad_copy(Model& source) {

		for (int i = 0; i < layer_.size(); i++) {

			for (int j = 0; j < layer_[i].node_num_; j++) {

				layer_[i].grad_[j] = source.layer_[i].grad_[j];

				//output layer outofrange
				if (i == layer_.size() - 1) {
					break;
				}
				for (int l = 0; l < layer_[i + 1].node_num_; l++) {

					layer_[i].connection_.weight_grad_[j][l] = source.layer_[i].connection_.weight_grad_[j][l];
				}
			}
		}

		return;
	}

	vector <double> Model::predict() {
		return layer_.rbegin()->result_;
	}

	void Model::model_save(string filename)
	{
		FILE* fs;
		fs = fopen(filename.c_str(), "wb");
		if (fs == NULL) {
			printf("can't open file to write\n");
			return;
		}

		//hyper_parm
		fwrite(&hyper_parm_, sizeof(HyperParm), 1, fs);

		//layer_num
		int tmp = layer_.size();
		fwrite(&tmp, sizeof(int), 1, fs);

		//layers
		for (int i = 0; i < layer_.size(); i++)
		{
			fwrite(&layer_[i].node_num_, sizeof(int), 1, fs);
			fwrite(&layer_[i].activation_, sizeof(Activation), 1, fs);
		}

		//elements of layer
		for (int i = 0; i < layer_.size(); i++) {

			// layer::bias*
			for (int j = 0; j < layer_[i].node_num_; j++) {
				fwrite(&layer_[i].bias_[j], sizeof(double), 1, fs);
			}

			//output_layer out_of_range
			if (i == layer_.size() - 1) {
				break;
			}

			// layer::connection
			for (int col = 0; col < layer_[i].node_num_; col++) {
				for (int row = 0; row < layer_[i + 1].node_num_; row++) {

					// layer::connection::weight**
					fwrite(&layer_[i].connection_.weight_[col][row], sizeof(double), 1, fs);
				}
			}
		}

		fclose(fs);

		return;
	}

	void Model::model_load(string filename)
	{
		FILE* fs;
		fs = fopen(filename.c_str(), "rb");
		if (fs == NULL) {
			printf("can't open file to read\n");
			return;
		}

		//clear
		layer_.clear();

		//hyper_parm
		fread(&hyper_parm_, sizeof(HyperParm), 1, fs);

		//layer_num
		int layer_num;
		fread(&layer_num, sizeof(int), 1, fs);

		//layers
		layer_ = vector<Layer>(layer_num);
		for (int i = 0; i < layer_num; i++) {

			fread(&layer_[i].node_num_, sizeof(int), 1, fs);
			fread(&layer_[i].activation_, sizeof(Activation), 1, fs);
		}

		//resize layers & elements
		init(true);

		//elements of layer
		for (int i = 0; i < layer_num; i++) {

			// layer::bias*
			for (int j = 0; j < layer_[i].node_num_; j++) {
				fread(&layer_[i].bias_[j], sizeof(double), 1, fs);
			}

			//output_layer out_of_range
			if (i == layer_num - 1) {
				break;
			}

			// layer::connection
			for (int col = 0; col < layer_[i].node_num_; col++) {
				for (int row = 0; row < layer_[i + 1].node_num_; row++) {
					
					// layer::connection::weight**
					fread(&layer_[i].connection_.weight_[col][row], sizeof(double), 1, fs);
				}
			}
		}

		return;
	}

	//TODO
}