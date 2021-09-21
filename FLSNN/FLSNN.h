#pragma once

#include <stdio.h>
#include <vector>
#include <thread>
#include <string>
using namespace std;

class Layer {
private:
public:
	//Parametor
	int node_num_;
	string activation_;
	vector<Layer*> next_; ///< Pointer next layer (can multiple)
	vector<vector<double>> weight_;
	vector<vector<double>> stochastic_gate_rate_;
	vector<double> bias_;
	vector<double> result_;

	//Backprop
	vector<vector<double>> weight_grad_;
	vector<vector<int>> stochastic_gate_grad_;
	vector<double> grad_;

	//Function
	void build(); ///< Build a layer
};

class FLNN {
private:
	int step; ///< forward step
public:
	/*HyperParametor*/
	//essential
	int input_node_num_;
	int hidden_num_;
	int hidden_node_num_;
	int output_node_num_;
	//optional
	double learning_rate_;
	double grad_clipping_;
	double backprop_truncate_step_; ///< Backprop �ݺ� �ִ� Ƚ�� ����
	string activation_;
	string loss_;
	double stochastic_rate_init_; ///< stochastic_gate init value
	double bias_init_; ///< bias init value

	//Function
	void build(); ///< Build the model
	void forward_step(vector<double> input); ///< �� ��ü ������ ����
	void forward_step(); ///< ���� input���� loop�� �����͸� ��ȸ��
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