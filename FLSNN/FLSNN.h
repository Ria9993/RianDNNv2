#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
using namespace std;

#include <thread>
#include <ppl.h>
using namespace concurrency;

namespace FLSNN {
	inline void ReLU(double *x) {
		*x = max((double)0, *x);
		return;
	}

	class HyperParm {
	private:
	public:
		double learning_rate_;
		double grad_clipping_;
		double backprop_depth_limit_; ///< Backprop depth 제한 (Default : model_depth)
		string loss_;
		double stochastic_rate_init_; ///< stochastic_gate init value
		double bias_init_; ///< bias init value

		HyperParm() {
			//Default optional parametor set
			learning_rate_ = 0.001f;
			grad_clipping_ = 100.0f;
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
		vector<double> calc_result_;
		vector<double> result_;
		bool build_flag_;

		//Backprop Gradient
		vector<double> grad_;


		//Layer init
		Layer(int node_num, string activation) {
			node_num_ = node_num;
			activation_ = activation;
		}

		//Layer Connect
		Layer operator >> (Layer& x) {
			next_.push_back(&x);
			x.last_.push_back(this);
			return *this;
		}
	};

	class Iterator {
	private:
	public:
		//element
		vector<pair<Layer*, Layer*>> route_;
		vector<Layer*> list_; ///< 중복 없는 layer_list
		int execute_num_; //run 횟수

		Iterator() {
			execute_num_ = 0;
		}

		//Function
		void add(Layer* source, Layer* dest);
		void build(HyperParm* hyper_parm);
		void build(Layer* layer, HyperParm* hyper_parm);
		void run();
		void calc(Layer* source, Layer* dest);
		///todo prediect
	};

	void Iterator::add(Layer* source, Layer* dest) {
		*source >> *dest;
		route_.push_back({ source,dest });

		bool s_flag = false, d_flag = false;
		for (int i = 0; i < list_.size(); i++) {
			if (source == list_[i])
				s_flag = true;
			if (dest == list_[i])
				d_flag = true;
		}
		if (s_flag == false)
			list_.push_back(source);
		if (d_flag == false)
			list_.push_back(dest);

		return;
	}

	void Iterator::build(HyperParm* hyper_parm) {
		for (int i = 0; i < route_.size(); i++) {
			build(route_[i].first, hyper_parm);
			build(route_[i].second, hyper_parm);
		}
		return;
	}

	void Iterator::build(Layer* layer, HyperParm* hyper_parm) {
		//Check flag
		if (layer->build_flag_ == true)
			return;
		else {
			layer->build_flag_ = true;
			layer->execute_num_ = 0;
		}

		//element init
		layer->bias_.resize(layer->node_num_, hyper_parm->bias_init_);
		layer->calc_result_.resize(layer->node_num_, 0);
		layer->result_.resize(layer->node_num_, 0);
		layer->grad_.resize(layer->node_num_, 0);

		//connection init
		random_device rd;
		mt19937 gen(rd());
		normal_distribution<double> HE(0, (double)2 / layer->node_num_); ///< HE initialization
		layer->connection_.resize(layer->next_.size());
		for (int i = 0; i < layer->next_.size(); i++) {
			layer->connection_[i].weight_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_));
			layer->connection_[i].weight_grad_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_, 0));
			layer->connection_[i].stochastic_gate_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_));
			layer->connection_[i].stochastic_gate_grad_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_, 0));
			//Weight, stochastic_gate init
			for (int j = 0; j < layer->node_num_; j++) {
				for (int k = 0; k < layer->next_[i]->node_num_; k++) {
					layer->connection_[i].weight_[j][k] = HE(gen);
					layer->connection_[i].stochastic_gate_[j][k] = hyper_parm->stochastic_rate_init_;
				}
			}
		}

		return;
	}

	void Iterator::run() {
		//reset calc_result
		for (int i = 0; i < list_.size(); i++) {
			Layer* layer = list_[i];
			layer->calc_result_.assign(layer->calc_result_.size(), 0);
		}

		//calc by route
		for (int i = 0; i < route_.size(); i++) {
			calc(route_[i].first, route_[i].second);
		}

		execute_num_++;
	}

	void Iterator::calc(Layer* source, Layer* dest) {
		random_device rd;
		mt19937 gen(rd());

		//find dest_index of source
		int dest_idx;
		for (int i = 0; i < source->next_.size(); i++) {
			if (source->next_[i] == dest) {
				dest_idx = i;
			}
		}

		//copy calc_result to result
		source->result_ = source->calc_result_;

		//Parallel calc
		parallel_for(0, dest->node_num_, [&](int n) {
			//Activation
			if (source->activation_ == "ReLU")
				ReLU(&source->calc_result_[n]);
			if (source->activation_ == "Sigmoid");
			else;

			for (int j = 0; j < source->node_num_; j++) {
				//Stochastic gate
				uniform_real_distribution<double> rnd(0, 1);
				if (rnd(gen) > source->connection_[dest_idx].stochastic_gate_[j][n]) {
					//Weight
					dest->calc_result_[n] += source->result_[j] * source->connection_[dest_idx].weight_[j][n];
				}
			}
			});

		return;
	}

	//TODO
	//model save & load
}