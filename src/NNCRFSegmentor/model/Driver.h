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
  Driver() {
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


public:
	//embeddings are initialized before this separately.
	inline void initial() {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_builder.resize(_hyperparams.batch);
		for(int idx = 0; idx < _hyperparams.batch; idx++) {
			_builder[idx].createNodes(GraphBuilder::max_sentence_length, _modelparams.types.size());
			_builder[idx].initial(&_cg, _modelparams, _hyperparams);
		}
		
		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}

	inline void directInitial() {
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_builder.resize(1);
		_builder[0].createNodes(GraphBuilder::max_sentence_length, _modelparams.types.size());
		_builder[0].initial(&_cg, _modelparams, _hyperparams);
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
			int seq_size = example.m_features.size();
			cost += _modelparams.loss.loss(getPNodes(_builder[count].output, seq_size), example.m_labels, _eval, example_num);
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
		_cg.compute();
		_modelparams.loss.predict(getPNodes(_builder[0].output, seq_size), results);
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
		cost += _modelparams.loss.cost(getPNodes(_builder[0].output, seq_size), example.m_labels, 1);

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

	void saveModel(std::ofstream& os) const {
		_hyperparams.saveModel(os);
		_modelparams.saveModel(os);
	}

	void loadModel(std::ifstream& is) {
		_hyperparams.loadModel(is);
		_modelparams.loadModel(is);
	}


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
