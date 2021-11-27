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

	class HyperParm {
	private:
	public:
		double learning_rate_;
		double grad_clipping_;
		//double momentum_rate_; ///< Optimizor momentum
		double backprop_depth_limit_; ///< Backprop depth 제한
		double momentum_rate_;
		string loss_;
		double backprop_rate_; ///< network exploding 방지(develop)
		double stochastic_rate_init_; ///< stochastic_gate init value
		double bias_init_; ///< bias init value

		HyperParm() {
			//Default optional parametor set
			learning_rate_ = 0.1e-2f;
			grad_clipping_ = 100.0f;
			backprop_depth_limit_ = 100;
			momentum_rate_ = 0.8f;
			loss_ = "MSE";
			backprop_rate_ = 0.66f; ///< develop
			stochastic_rate_init_ = -1.0;
			bias_init_ = 0.01;
		}
	};

	class Connection {
	private:
	public:
		////Pointer
		///Layer* dest_;
		///Layer* source_;

		//Element
		vector<vector<double>> weight_;
		vector<vector<double>> stochastic_gate_;

		//Backprop Gradient
		vector<vector<double>> weight_grad_;
		vector<vector<double>> weight_grad_momentum_;
		vector<vector<double>> stochastic_gate_grad_;
		vector<vector<double>> stochastic_gate_grad_momentum_;
	};

	enum class Activation {
		None,
		ReLU,
		Sigmoid
	};

	inline double derivative(Activation activation, double x) {
		switch (activation) {
		case Activation::ReLU :
			if (x <= 0)
				return 0;
			else
				return 1;
			break;
		case Activation::Sigmoid :
			break;
		case Activation::None :
			return 1;
			break;
		default :
			break;
		}

		return -1;
	}

	class Layer {
	private:
	public:
		//Parametor
		int node_num_;
		Activation activation_;

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
		int backprop_done_;

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

		//Layer Connect
		Layer* operator >> (Layer& x) {
			next_.push_back(&x);
			x.last_.push_back(this);
			return this;
		}
	};

	class Iterator {
	private:
	public:
		//element
		vector<pair<Layer*, Layer*>> route_;
		vector<Layer*> list_; ///< 중복 없는 layer_list
		Layer* output_; ///< pointer of output layer
		HyperParm* hyper_parm_; ///< pointer of hyper_parm
		int execute_num_; ///< run 횟수
		double loss_;

		Iterator() {
			execute_num_ = 0;
			loss_ = 0;
		}
		Iterator(HyperParm *hyper_parm) {
			hyper_parm_ = hyper_parm;
			execute_num_ = 0;
			loss_ = 0;
		}

		//Function
		void add(Layer* source, Layer* dest);
		void init(); ///< resize & init elements
		void init(bool load_flag); ///< just resize elements
		void init(Layer* layer, bool load_flag);
		void run(vector<double>& input, vector<double>& target);
		void run(vector<double>& input);
		void calc(Layer* source, Layer* dest);
		void optimize();
		void backprop(Layer* layer, Layer* source, int depth);
		void grad_clear();
		void model_save(); ///< file save & load
		void model_load();
		vector <double> predict();

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

	void Iterator::init() {
		for (int i = 0; i < list_.size(); i++) {
			init(list_[i], false);
		}
		return;
	}

	void Iterator::init(bool load_flag) {
		for (int i = 0; i < route_.size(); i++) {
			init(route_[i].first, load_flag);
			init(route_[i].second, load_flag);
		}
		return;
	}

	void Iterator::init(Layer* layer, bool load_flag) {

		//element init
		layer->execute_num_ = 0;
		layer->backprop_done_ = 0;
		layer->bias_.resize(layer->node_num_, hyper_parm_->bias_init_);
		layer->calc_result_.resize(layer->node_num_, 0);
		layer->result_.resize(layer->node_num_, 0);
		layer->grad_.resize(layer->node_num_, 0);
		layer->grad_momentum_.resize(layer->node_num_, 0);

		//connection init
		random_device rd;
		mt19937 gen(rd());

		layer->connection_.resize(layer->next_.size());
		for (int i = 0; i < layer->next_.size(); i++) {

			layer->connection_[i].weight_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_));
			layer->connection_[i].weight_grad_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_, 0));
			layer->connection_[i].weight_grad_momentum_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_, 0));

			layer->connection_[i].stochastic_gate_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_));
			layer->connection_[i].stochastic_gate_grad_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_, 0));
			layer->connection_[i].stochastic_gate_grad_momentum_.resize(layer->node_num_, vector<double>(layer->next_[i]->node_num_, 0));

			//Weight, stochastic_gate init
			if (load_flag == false) { ///< 모델 불러오기 시 초기화 배제
				
				normal_distribution<double> HE(0, sqrtf((double)2 / layer->node_num_)); ///< HE initialization
				for (int j = 0; j < layer->node_num_; j++) {
					for (int k = 0; k < layer->next_[i]->node_num_; k++) {

						layer->connection_[i].weight_[j][k] = HE(gen);
						layer->connection_[i].stochastic_gate_[j][k] = hyper_parm_->stochastic_rate_init_;
					}
				}
			}
		}

		return;
	}

	void Iterator::run(vector<double>& input) {
		
		//input set
		list_[0]->result_ = input;

		//reset calc_result
		for (int i = 0; i < list_.size(); i++) {
			Layer* layer = list_[i];
			layer->calc_result_ = layer->bias_;
		}

		//calc by route
		for (int i = 0; i < route_.size(); i++) {
			calc(route_[i].first, route_[i].second);
		}
		
		return;
	}

	void Iterator::run(vector<double>& input, vector<double>& target) {

		//input set
		list_[0]->result_ = input;

		//reset calc_result
		for (int i = 0; i < list_.size(); i++) {
			Layer* layer = list_[i];
			layer->calc_result_ = layer->bias_;
		}

		//calc by route
		for (int i = 0; i < route_.size(); i++) {
			calc(route_[i].first, route_[i].second);
		}

		//calc derivative of loss
		double last_loss = loss_;
		loss_ = 0;
		for (int i = 0; i < output_->node_num_; i++) {
			double tmp = 0;

			if (hyper_parm_->loss_ == "MSE") {
				tmp = output_->result_[i] - target[i];
				///output_->grad_[i] += 2 * fabs(tmp);
				output_->grad_[i] += 2 * tmp; ///< derivative of loss
				loss_ += tmp * tmp;
			}
			else;
		}
		loss_ /= output_->node_num_;

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

		//Activation
		for (int i = 0; i < source->node_num_; i++) {
			switch (source->activation_) {
			case Activation::ReLU :
				source->result_[i] = fmax(0, source->result_[i]);
				//source->grad_[i] += derivative(Activation::ReLU, source->result_[i]);
				break;
			//case Activation::LeakyReLU :
			//	source->result_[i] = fmax(source->result_[i] * 0.01, source->result_[i]);
			//	break;
			case Activation::Sigmoid : 
				break;
			case Activation::None :
				//source->grad_[i] += 1;
				break;
			default: 
				break;
			}
		}

		//multi-threaded Calculate
		parallel_for(0, dest->node_num_, [&](int n) {
			for (int j = 0; j < source->node_num_; j++) {

				//Weight
				dest->calc_result_[n] += source->result_[j] * source->connection_[dest_idx].weight_[j][n];
				//Gradient
				source->connection_[dest_idx].weight_grad_[j][n] += source->result_[j];
				source->connection_[dest_idx].stochastic_gate_grad_[j][n] += 1;
			}
		});



		//copy calc_result to result
		dest->result_ = dest->calc_result_;

		return;
	}

	void Iterator::optimize()
	{
		//backprop
		for (int i = 0; i < output_->last_.size(); i++) {
			backprop(output_->last_[i], output_, 1);
		}

		grad_clear();
		return;
	}

	void Iterator::backprop(Layer* layer, Layer* source, int depth)
	{
		//check for backprop_depth_limit
		if (depth >= hyper_parm_->backprop_depth_limit_)
			return;

		//find source_index of layer
		int source_idx;
		for (int i = 0; i < layer->next_.size(); i++) {
			if (layer->next_[i] == source) {
				source_idx = i;
			}
		}

		//calc grad & elements update
		for (int i = 0; i < layer->node_num_; i++) {
			for (int j = 0; j < source->node_num_; j++) {

				double grad_tmp;
				//weight
				layer->connection_[source_idx].weight_grad_momentum_[i][j] *= hyper_parm_->momentum_rate_;
				grad_tmp = (source->grad_[j] * layer->connection_[source_idx].weight_grad_[i][j]) / execute_num_;
				layer->connection_[source_idx].weight_grad_momentum_[i][j] += grad_tmp;
				layer->connection_[source_idx].weight_[i][j] -= hyper_parm_->learning_rate_ * layer->connection_[source_idx].weight_grad_momentum_[i][j];

				//stochastic_gate
				layer->connection_[source_idx].stochastic_gate_grad_momentum_[i][j] *= hyper_parm_->momentum_rate_;
				grad_tmp *= layer->connection_[source_idx].stochastic_gate_grad_[i][j] / execute_num_;
				layer->connection_[source_idx].stochastic_gate_grad_momentum_[i][j] += grad_tmp;
				layer->connection_[source_idx].stochastic_gate_[i][j] -= hyper_parm_->learning_rate_ * layer->connection_[source_idx].stochastic_gate_grad_momentum_[i][j];

				//backprop
				layer->grad_[i] += grad_tmp;
			}
		}

		layer->backprop_done_++;

		//wait for backprop_chain
		if (layer->backprop_done_ == layer->next_.size()) {

			//bias update
			for (int i = 0; i < layer->node_num_; i++) {
				layer->grad_momentum_[i] *= hyper_parm_->momentum_rate_;
				layer->grad_momentum_[i] += layer->grad_[i] / execute_num_;
				layer->bias_[i] -= hyper_parm_->learning_rate_ * layer->grad_momentum_[i];
			}

			//backprop recursive
			for (int i = 0; i < layer->last_.size(); i++) {
				backprop(layer->last_[i], layer, depth + 1);
			}
		}

		return;
	}

	void Iterator::grad_clear()
	{
		for (int i = 0; i < list_.size(); i++) {
			list_[i]->backprop_done_ = 0;

			for (int j = 0; j < list_[i]->node_num_; j++) {
				list_[i]->grad_[j] = 0;

				for (int k = 0; k < list_[i]->next_.size(); k++) {
					for (int l = 0; l < list_[i]->next_[k]->node_num_; l++) {

						list_[i]->connection_[k].weight_grad_[j][l] = 0;
						list_[i]->connection_[k].stochastic_gate_grad_[j][l] = 0;
					}
				}
			}
		}

		execute_num_ = 0;
		return;
	}

	vector <double> Iterator::predict() {
		return output_->result_;
	}

	/* @ Model save & load Format
	hyper_parm
	layer_num
	-Layer * [layer_num] {
		node_num
		activation
	}
	route_num
	route * [route_num] pair<int,int>
	outupt_idx
	-Layer * [layer_num] {
		bias * [node_num]
		-Connection ** [next_size] {
			weight ** [node_num * next_node_num]
			stochastic_gate ** [node_num * next_node_num]
		}
	}
	*/
	void Iterator::model_save()
	{
		FILE* fs;
		fs = fopen("model.data", "wb");
		if (fs == NULL) {
			printf("can't open file to write\n");
			return;
		}

		//hyper_parm
		fwrite(hyper_parm_, sizeof(HyperParm), 1, fs);

		//layer_num
		int tmp = list_.size();
		fwrite(&tmp, sizeof(int), 1, fs);

		//layers
		for (int i = 0; i < list_.size(); i++)
		{
			fwrite(&list_[i]->node_num_, sizeof(int), 1, fs);
			fwrite(&list_[i]->activation_, sizeof(Activation), 1, fs); ///< 짧아서 string자체로 저장해도 됨
		}

		//route_num
		tmp = route_.size();
		fwrite(&tmp, sizeof(int), 1, fs);

		//route
		///find layer_idx by pointer
		for (int i = 0; i < route_.size(); i++) {
			int f_idx, s_idx;

			for (int j = 0; j < list_.size(); j++) {
				if (list_[j] == route_[i].first)
					f_idx = j;
				if (list_[j] == route_[i].second)
					s_idx = j;
			}
			fwrite(&f_idx, sizeof(int), 1, fs);
			fwrite(&s_idx, sizeof(int), 1, fs);
		}

		//output_idx
		int output_idx;
		for (int i = 0; i < list_.size(); i++) {
			if (output_ == list_[i])
				output_idx = i;
		}
		fwrite(&output_idx, sizeof(int), 1, fs);

		//elements of layer
		for (int i = 0; i < list_.size(); i++) {

			// layer::bias*
			for (int j = 0; j < list_[i]->node_num_; j++) {
				fwrite(&list_[i]->bias_[j], sizeof(double), 1, fs);
			}

			// layer::connection*
			for (int j = 0; j < list_[i]->next_.size(); j++) {
				for (int col = 0; col < list_[i]->node_num_; col++) {
					for (int row = 0; row < list_[i]->next_[j]->node_num_; row++) {
						// layer::connection::weight**
						fwrite(&list_[i]->connection_[j].weight_[col][row], sizeof(double), 1, fs);
						// layer::connection::stochastic_gate**
						fwrite(&list_[i]->connection_[j].stochastic_gate_[col][row], sizeof(double), 1, fs);
					}
				}
			}
		}

		fclose(fs);

		return;
	}

	void Iterator::model_load()
	{
		FILE* fs;
		fs = fopen("model.data", "rb");
		if (fs == NULL) {
			printf("can't open file to read\n");
			return;
		}

		//clear Iterator
		route_.clear();
		list_.clear();
		//memset(&route_, NULL, sizeof(vector<pair<Layer*, Layer*>>));
		//memset(&list_, NULL, sizeof(vector<Layer*>));

		//hyper_parm
		fread(hyper_parm_, sizeof(HyperParm), 1, fs);

		//layer_num
		int layer_num;
		fread(&layer_num, sizeof(int), 1, fs);

		//layers
		static vector<Layer> layer(layer_num);
		for (int i = 0; i < layer_num; i++) {
			//list add
			list_.push_back(&layer[i]);

			fread(&layer[i].node_num_, sizeof(int), 1, fs);
			fread(&layer[i].activation_, sizeof(Activation), 1, fs);

			//vector는 원소까지 로드가 안되므로 초기화
			//memset(&layer[i].next_, NULL, sizeof(vector<Layer*>));
			//memset(&layer[i].last_, NULL, sizeof(vector<Layer*>));
			//memset(&layer[i].connection_, NULL, sizeof(vector<Connection>));
			//memset(&layer[i].bias_ , NULL, sizeof(vector<double>));
			//memset(&layer[i].calc_result_, NULL, sizeof(vector<double>));
			//memset(&layer[i].result_, NULL, sizeof(vector<double>));
			//memset(&layer[i].grad_, NULL, sizeof(vector<double>));
			//memset(&layer[i].grad_momentum_, NULL, sizeof(vector<double>));
		}

		//route_num
		int route_num;
		fread(&route_num, sizeof(int), 1, fs);

		//route
		int source, dest;
		for (int i = 0; i < route_num; i++) {
			fread(&source, sizeof(int), 1, fs);
			fread(&dest, sizeof(int), 1, fs);
			//route_.push_back({ &layer[source], &layer[dest] });
			add(&layer[source], &layer[dest]);
		}

		//output_idxvector<pair<Layer*, Layer*>>
		int output_idx;
		fread(&output_idx, sizeof(int), 1, fs);
		output_ = &layer[output_idx];

		//resize layers & elements
		init(true);

		//elements of layer
		for (int i = 0; i < list_.size(); i++) {

			// layer::bias*
			for (int j = 0; j < list_[i]->node_num_; j++) {
				fread(&list_[i]->bias_[j], sizeof(double), 1, fs);
			}

			// layer::connection*
			for (int j = 0; j < list_[i]->next_.size(); j++) {
				for (int col = 0; col < list_[i]->node_num_; col++) {
					for (int row = 0; row < list_[i]->next_[j]->node_num_; row++) {
						// layer::connection::weight**
						fread(&list_[i]->connection_[j].weight_[col][row], sizeof(double), 1, fs);
						// layer::connection::stochastic_gate**
						fread(&list_[i]->connection_[j].stochastic_gate_[col][row], sizeof(double), 1, fs);
					}
				}
			}
		}

		return;
	}

	//TODO
}