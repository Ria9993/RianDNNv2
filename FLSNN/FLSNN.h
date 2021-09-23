#pragma once

#include <stdio.h>
#include <vector>
#include <thread>
#include <string>
using namespace std;

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
	vector<Connection> next_;
	vector<Layer*> last_;

	//Element
	vector<double> bias_;
	vector<double> result_;

	//Backprop Gradient
	vector<double> grad_;

	//Function
	void build(); ///< Build a layer
};

class FLNN {
private:
	int step; ///< forward step
public:
	//HyperParam
	///essential
	int input_node_num_;
	int hidden_num_;
	int hidden_node_num_;
	int output_node_num_;
	///optional
	double learning_rate_;
	double grad_clipping_;
	double backprop_truncate_step_; ///< Backprop 반복 최대 횟수 설정
	string activation_;
	string loss_;
	double stochastic_rate_init_; ///< stochastic_gate init value
	double bias_init_; ///< bias init value

	//Function
	void build(); ///< Build the model
	void forward_step(vector<double> input); ///< 한 전체 루프를 진행
	void forward_step(); ///< 없는 input으로 loop된 데이터만 공회전
	void optimize(vector<double> target);
	void reset_grad();
	void reset();

	FLNN() {
		//Default optional parametor set
		learning_rate_ = 0.001f;
		grad_clipping_ = 100.0f;
		backprop_truncate_step_ = 2;
		activation_ = "ReLU";
		loss_ = "MSE";
		stochastic_rate_init_ = 0.3f;
		bias_init_ = 0.01f;

		//Reset variable
		step = 0;
	}
};

inline void FLNN::build()
{

}

inline void FLNN::forward_step(vector<double> input)
{
}

inline void FLNN::forward_step()
{
	vector<double> empty_input;
	empty_input.resize(input_node_num_);
	memset(&empty_input, 0, sizeof(empty_input));

	forward_step(empty_input);

	return;
}

inline void FLNN::optimize(vector<double> target)
{
}

inline void FLNN::reset_grad()
{
}

inline void FLNN::reset()
{
}