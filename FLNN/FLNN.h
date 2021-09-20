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
	
	//Backprop
	vector<vector<double>> weight_grad_;
	vector<vector<int>> stochastic_gate_grad_;
	vector<double> grad_;

	//Function
	void make() ///< Build a layer
};

class FLNN {
private:
	int step;
public:
	/*HyperParametor*/
	//essential
	int input_node_num_;
	int hidden_num_;
	int hidden_node_num_;
	int output_node_num_;
	//optional
	double learning_rate_;
	double clipping_threshold_;
	double backprop_truncate_step_; ///< Backprop 반복 최대 횟수 설정
	string activation_;
	string loss_;
	double stochastic_rate_; ///< 신경 전달 확률

	//Function
	void make(); ///< Build & make the model
	void forward_step(vector<double> input); ///< 한 전체 루프를 진행
	void forward_step(); ///< 없는 input으로 loop된 데이터만 공회전
	void optimize(vector<double> target);
	void reset();

	FLNN() {
		//Default optional parametor set
		learning_rate_ = 0.001f;
		clipping_threshold_ = 100.0f;
		backprop_truncate_step_ = 2;
		activation_ = "ReLU";
		loss_ = "MSE";
		stochastic_rate_ = 0.3f;

		//Reset variable
		step = 0;
	}
};

inline void FLNN::make()
{

}

inline void FLNN::forward_step(vector<double> input)
{
}

inline void FLNN::forward_step()
{
	vector<double> empty_input;
	empty_input.resize(input_node_num_);
	memset(empty_input, 0, sizeof(empty_input));

	forward_step(empty_input)

	return;
}

inline void FLNN::optimize(vector<double> target)
{
}

inline void FLNN::reset()
{
}