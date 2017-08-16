/*
 * SegOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SEGBuilder_H_
#define SEGBuilder_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Pooling.h"
#include "UniOP.h"
#include "TriOP.h"

struct SegParams {
	UniParams H;
	TriParams merge;
	int inDim;
	int outDim;
	int hiddenDim;

	SegParams() {		
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		H.exportAdaParams(ada);
		merge.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nHSize, int nISize) {
		H.initial(nHSize, nISize, true);
		merge.initial(nOSize, nHSize, nHSize, nHSize, true);
		inDim = nISize;
		outDim = nOSize;
		hiddenDim = nHSize;		
	}

	inline void save(std::ofstream &os) const {
		H.save(os);
		merge.save(os);
		os << inDim << endl;
		os << outDim << endl;
		os << hiddenDim << endl;
	}

	inline void load(std::ifstream &is) {
		H.load(is);
		merge.load(is);
		is >> inDim;
		is >> outDim;
		is >> hiddenDim;
	}
};

// we can rewrite it as one node, but many duplicated codes
class SegBuilder {
public:
	SegParams* _param;

	int _nSize;
	int _inDim;
	int _outDim;
	int _hiddenDim;

	TriNode _output;
	SumPoolNode _sum;
	MaxPoolNode _max;
	MinPoolNode _min;
	vector<UniNode> _tnodes;

public:
	SegBuilder(){
		clear();
	}

	~SegBuilder(){
		clear();
	}

	inline void init(SegParams* paramInit, dtype dropout) {
		_param = paramInit;
		_inDim = _param->inDim;
		_outDim = _param->outDim;
		_hiddenDim = _param->hiddenDim;
		_output.setParam(&_param->merge);
		_output.init(_outDim, dropout);
		_sum.setParam(_hiddenDim);
		_sum.init(_hiddenDim, -1);
		_max.setParam(_hiddenDim);
		_max.init(_hiddenDim, -1);
		_min.setParam(_hiddenDim);
		_min.init(_hiddenDim, -1);

		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes[idx].setParam(&(_param->H));
			_tnodes[idx].init(_hiddenDim, dropout);
		}
	}

	inline void setFunctions(dtype(*f)(const dtype&),
		dtype(*f_deri)(const dtype&, const dtype&)) {
		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes[idx].setFunctions(f, f_deri);
		}
		_output.setFunctions(f, f_deri);
	}

	inline void resize(int maxsize){
		_tnodes.resize(maxsize);
	}

	inline void clear(){
		_tnodes.clear();
		_param = NULL;
		_inDim = 0;
		_outDim = 0;
		_hiddenDim = 0;
		_nSize = 0;
	}

public:

	inline void forward(Graph *cg, const vector<PNode>& x){
		if (x.size() == 0){
			std::cout << "empty inputs for seg operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.dim != _inDim){
			std::cout << "input dim does not match for seg operation" << std::endl;
			return;
		}

		for (int idx = 0; idx < _nSize; idx++){
			_tnodes[idx].forward(cg, x[idx]);
		}

		_sum.forward(cg, getPNodes(_tnodes, _nSize));
		_max.forward(cg, getPNodes(_tnodes, _nSize));
		_min.forward(cg, getPNodes(_tnodes, _nSize));
		_output.forward(cg, &_sum, &_max, &_min);
	}

};

#endif /* SEGBuilder_H_ */
