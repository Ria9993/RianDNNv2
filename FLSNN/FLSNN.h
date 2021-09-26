#pragma once

#include <stdio.h>
#include <vector>
#include <thread>
#include <string>
#include <random>
using namespace std;

namespace FLSNN {
	class HyperParm {
	private:
	public:
		int execute_num_;
		double learning_rate_;
		double grad_clipping_;
		double backprop_depth_limit_; ///< Backprop depth ���� (Default : model_depth)
		string loss_;
		double stochastic_rate_init_; ///< stochastic_gate init value
		double bias_init_; ///< bias init value

		HyperParm() {
			//Default optional parametor set
			learning_rate_ = 0.001f;
			grad_clipping_ = 100.0f;
			execute_num_ = 0;
			loss_ = "MSE";
			stochastic_rate_init_ = 0.3f;
			bias_init_ = 0.01f;
		}
	};

	class Connection {
	private:
	public:
		////Pointer
		//Layer* dest_;
		//Layer* source_;

		//Element
		vector<vector<double>> weight_;
		vector<vector<double>> stochastic_gate_;

		//Backprop Gradient
		vector<vector<double>> weight_grad_;
		vector<vector<double>> stochastic_gate_grad_;
	};

	class Layer {
	private:
	public:
		//Parametor
		int node_num_;
		string activation_;

		//Connection
		vector<Layer*> next_;
		vector<Connection> connection_;
		vector<Layer*> last_;

		//Element
		int execute_num_;
		vector<double> bias_;
		vector<double> result_;
		bool build_flag_;

		//Backprop Gradient
		vector<double> grad_;

		//Function
		void build(HyperParm* parm); ///< Build layer_chain & Auto set Hyperparm

		//Layer init
		Layer(int node_num, string activation) {
			node_num_ = node_num;
			activation_ = activation;
		}

		//Layer Connect
		Layer operator += (Layer& x) {
			next_.push_back(&x);
			x.last_.push_back(this);
			return *this;
		}
	};

	void Layer::build(HyperParm* parm)
	{
		if (build_flag_ == true)
			return;
		else
			build_flag_ = true;

		//element init
		bias_.resize(node_num_, parm->bias_init_);
		result_.resize(node_num_, 0);
		grad_.resize(node_num_, 0);

		//connection init
		random_device rd;
		mt19937 gen(rd());
		normal_distribution<double> HE(0, (double)2 / this->node_num_); ///< HE initialization
		connection_.resize(next_.size());
		for (int i = 0; i < next_.size(); i++) {
			connection_[i].weight_.resize(this->node_num_, vector<double>(next_[i]->node_num_));
			connection_[i].weight_grad_.resize(this->node_num_, vector<double>(next_[i]->node_num_, 0));
			connection_[i].stochastic_gate_.resize(this->node_num_, vector<double>(next_[i]->node_num_));
			connection_[i].stochastic_gate_grad_.resize(this->node_num_, vector<double>(next_[i]->node_num_, 0));
			//Weight HE init
			for (int j = 0; j < this->node_num_; j++) {
				for (int k = 0; k < next_[i]->node_num_; k++) {
					connection_[i].weight_[j][k] = HE(gen);
				}
			}
		}

		//Build chain
		for (int i = 0; i < next_.size(); i++) {
			next_[i]->build(parm);
		}
		return;
	}

	//TODO
	//model save & load
}