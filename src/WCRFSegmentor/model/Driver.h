/*
 * Driver.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"


//A native neural network classfier using only word embeddings

class Driver{
public:
  Driver(int memsize) : aligned_mem(memsize){
  }

	~Driver() {
	}

public:
	Graph _cg;  // build neural graphs
	vector<GraphBuilder> 	_builder;
	ModelParams _modelparams;  // model parameters
	HyperParams _hyperparams;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update

	AlignedMemoryPool aligned_mem;

public:
	//embeddings are initialized before this separately.
	inline void initial() {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams, &aligned_mem)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_builder.resize(_hyperparams.batch);
		for(int idx = 0; idx < _hyperparams.batch; idx++) {
			_builder[idx].createNodes(GraphBuilder::max_sentence_length);
			_builder[idx].initial(&_cg, _modelparams, _hyperparams, &aligned_mem);
		}
		
		std::cout << "allocated memory: " << aligned_mem.capacity << ", total required memory: " << aligned_mem.required << ", perc = " << aligned_mem.capacity*1.0 / aligned_mem.required << std::endl;

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();
		_cg.clearValue();
		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_builder[count].forward(example.m_features, true); 

			//loss function
			//for (int idx = 0; idx < seq_size; idx++) {
				//cost += _loss.loss(&(_cg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}

			// backward, which exists only for training 
		}
		_cg.compute();
		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];
			_builder[count].forward(example.m_features, true);
			int seq_size = example.m_features.size();
			//cost += _modelparams.loss.loss(getPNodes(_builder[count].output, seq_size), example.m_labels, _eval, example_num);
			for(int idx = 0; idx < seq_size; idx++)
				cost += _modelparams.loss.loss(&_builder[count].output[idx], example.m_labels[idx], _eval, example_num);
		}
		_cg.backward();

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, vector<int>& results) {
		_cg.clearValue();
		_builder[0].forward(features);
		int seq_size = features.size();
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_cg->output[idx]), results[idx]);
		//}
		results.resize(seq_size);
		_cg.compute();
		//_modelparams.loss.predict(getPNodes(_builder[0].output, seq_size), results);
		for (int idx = 0; idx < seq_size; idx++)
			_modelparams.loss.predict(&_builder[0].output[idx], results[idx]);
	}

	inline dtype cost(const Example& example){
		_cg.clearValue();
		_builder[0].forward(example.m_features); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_cg->output[idx]), example.m_labels[idx], 1);
		//}
		_cg.compute();
//		cost += _modelparams.loss.cost(getPNodes(_builder[0].output, seq_size), example.m_labels, 1);
		for(int idx = 0; idx < seq_size; idx++)
			cost += _modelparams.loss.loss(&_builder[0].output[idx], example.m_labels[idx], _eval, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);

	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



private:
	inline void resetEval() {
		_eval.reset();
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
