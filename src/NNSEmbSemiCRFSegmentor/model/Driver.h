#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "ComputionGraph.h"
#include <iostream>

class Driver {
public:
	Driver(int memsize) :_aligned_mem(memsize){
	}

	~Driver() {
	}

public:

	Metric _eval;

	ModelUpdate _ada;

	Graph _cg;

	vector<GraphBuilder> _builder;

	CheckGrad _checkgrad;

	ModelParams _model_params;

	HyperParams _hyper_params;

	AlignedMemoryPool _aligned_mem;

public:
	//embeddings are initialized before this separately.
	inline void initial(){
		if (!_hyper_params.bVaild()){
			std::cout << "hyper parameter initialization Error, please check!" << std::endl;
			return;
		}
		if (!_model_params.initial(_hyper_params)) {
			std::cout << "model parameter initialzation Errror, please check!" << std::endl;
			return;
		}
		_model_params.exportModelParams(_ada);
		_model_params.exportCheckGradParams(_checkgrad);

		_hyper_params.print();

		_builder.resize(_hyper_params.batch);
		for (int idx = 0; idx < _hyper_params.batch; idx++) {
			_builder[idx].createNodes(GraphBuilder::max_sentence_length, _hyper_params.maxsegLen, _model_params._types.size());
			_builder[idx].initial(&_cg, _model_params, _hyper_params, &_aligned_mem);
		}

		std::cout << "allocated memory: " << _aligned_mem.capacity << ", total required memory: " << _aligned_mem.required << ", perc = " << _aligned_mem.capacity*1.0 / _aligned_mem.required << std::endl;

		setUpdateParameters(_hyper_params.nnRegular, _hyper_params.adaAlpha, _hyper_params.adaEps);
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
		}
		_cg.compute();

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];
			int seq_size = example.m_features.size();
			cost += _model_params._loss.loss(_builder[count].poutput, example.m_seglabels, _eval, example_num);
		}
		_cg.backward();

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, NRMat<int>& results) {
		_cg.clearValue();
		_builder[0].forward(features);
		_cg.compute();
		int seq_size = features.size();
		_model_params._loss.predict(_builder[0].poutput, results);
	}

	inline dtype cost(const Example& example){
		_cg.clearValue();
		_builder[0].forward(example.m_features, true); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		_cg.compute();
		cost += _model_params._loss.cost(_builder[0].poutput, example.m_seglabels, 1);

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



public:
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
