#pragma once

#include <stdio.h>
#include <thread>
#include <vector>
using namespace std;

class FLNN {
public :
	int layer_num_;
	int layer_node_num_;
	int input_node_num_;
	int output_node_num_;
	
	double learning_rate_;
	double clipping_threshold_;

	void make();
	void forward_step();
	void optimize();
private :
	
	FLNN() {
		learning_rate_ = 0.001f;
		clipping_threshold_ = 100.0f;
	}
};