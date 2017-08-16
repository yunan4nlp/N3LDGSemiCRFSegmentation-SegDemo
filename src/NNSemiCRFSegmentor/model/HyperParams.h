#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
public:
	// must assign
	dtype dropProb;
	int wordContext;
	int hiddenSize1;
	int rnnHiddenSize;
	int hiddenSize2;
	int segHiddenSize;
	int inputSize;
	int maxsegLen;
	int labelSize;
	// auto generated
	int wordWindow;
	int wordDim;
	vector<int> typeDims;
	vector<int> maxLabelLength;
	int unitSize;
	int segLabelSize;
	int batch;
	
	// for optimization
	dtype nnRegular, adaAlpha, adaEps;

public:
	HyperParams(){
		bAssigned = false;
	}

	void setRequared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize1 = opt.hiddenSize;
		hiddenSize2 = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		segHiddenSize = opt.segHiddenSize;
		maxsegLen = opt.maxsegLen;
		dropProb = opt.dropProb;

		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		batch = opt.batchSize;

		bAssigned = true;
	}

	void clear() {
		bAssigned = false;
	}

	bool bVaild(){
		return bAssigned;
	}

	void print(){
		cout << "======HyperParams======" << endl;
		cout << "dropProb = " << dropProb << endl;
		cout << "wordContext = " << wordContext << endl;
		cout << "hiddenSize1 = " << hiddenSize1 << endl;
		cout << "rnnHiddenSize = " << rnnHiddenSize << endl;
		cout << "hiddenSize2 = " << hiddenSize2 << endl;
		cout << "segHiddenSize = " << segHiddenSize << endl;
		cout << "inputSize = " << inputSize << endl;
		cout << "maxsegLen = " << maxsegLen << endl;
		cout << "labelSize = " << labelSize << endl;
		cout << "wordWindow = "<< wordWindow << endl;
		cout << "wordDim = " << wordDim << endl;
		int typeDimSize = typeDims.size();
		for (int idx = 0; idx < typeDimSize; idx++)
			cout << "typeDims[" << idx << "] = " << typeDims[idx] << endl;
		int maxsize = maxLabelLength.size();
		for (int idx = 0; idx < maxsize; idx++)
			cout << "maxLabelLength[" << idx << "]= " << maxLabelLength[idx] << endl;
		cout << "unitSize = " << unitSize << endl;
		cout << "segLabelSize = " << segLabelSize << endl;
		cout << "batch = " << batch << endl;
		cout << "nnRegular = " << nnRegular << endl;
		cout << "adaAlpha = " << adaAlpha << endl;
		cout << "adaEps = " << adaEps << endl;
		cout << "======HyperParams======" << endl;
	}
	
	void saveModel(std::ofstream& os) const {
		os << dropProb << endl;
		os << wordContext << endl;
		os << hiddenSize1 << endl;
		os << rnnHiddenSize << endl;
		os << hiddenSize2 << endl;
		os << segHiddenSize << endl;
		os << inputSize << endl;
		os << maxsegLen << endl;
		os << labelSize << endl;
		os << wordWindow << endl;
		os << wordDim << endl;
		int typeDimSize = typeDims.size();
		os << typeDimSize << endl;
		for (int idx = 0; idx < typeDimSize; idx++)
			os << typeDims[idx] << endl;
		int maxsize = maxLabelLength.size();
		os << maxsize << endl;
		for (int idx = 0; idx < maxsize; idx++)
			os << maxLabelLength[idx] << endl;
		os << unitSize << endl;
		os << segLabelSize << endl;
		os << batch << endl;
		os << nnRegular << endl;
		os << adaAlpha << endl;
		os << adaEps << endl;
	}

	void loadModel(std::ifstream& is) {
		is >> dropProb;
		is >> wordContext;
		is >> hiddenSize1;
		is >> rnnHiddenSize;
		is >> hiddenSize2;
		is >> segHiddenSize;
		is >> inputSize;
		is >> maxsegLen;
		is >> labelSize;
		is >> wordWindow;
		is >> wordDim;
		int typeDimSize;
		is >> typeDimSize;
		typeDims.resize(typeDimSize);
		for (int idx = 0; idx < typeDimSize; idx++)
			is >> typeDims[idx];
		int maxsize;
		is >> maxsize;
		maxLabelLength.resize(maxsize);
		for (int idx = 0; idx < maxsize; idx++)
			is >> maxLabelLength[idx];
		is >> unitSize;
		is >> segLabelSize;
		is >> batch;
		is >> nnRegular;
		is >> adaAlpha;
		is >> adaEps;
	}
private:
	bool bAssigned;
};

#endif /* SRC_HyperParams_H_*/
