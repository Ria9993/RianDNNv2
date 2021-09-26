#pragma once

#include <stdio.h>
#include <vector>
#include <thread>
#include <string>
using namespace std;

namespace FLSNN {
	class HyperParm {
	private:
	public:
		double learning_rate_;
		double grad_clipping_;
		double backprop_depth_limit_; ///< Backprop depth มฆวั
		string loss_;
		double stochastic_rate_init_; ///< stochastic_gate init value
		double bias_init_; ///< bias init value

		HyperParm() {
			//Default optional parametor set
			//learning_rate_ = 0.001f;
			//grad_clipping_ = 100.0f;
			//backprop_depth_limit_ = 2;
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
		vector<vector<double>> stochastic_gate_rate_;

		//Backprop Gradient
		vector<vector<double>> weight_grad_;
		vector<vector<int>> stochastic_gate_grad_;
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
		Layer operator >> (Layer& x) {
			//Pointer set
			next_.push_back(&x);
			x.last_.push_back(this);
		}
	};

	void FLSNN::Layer::build(HyperParm* parm)
	{
		return;
	}

	//TODO
	//model save & load
}