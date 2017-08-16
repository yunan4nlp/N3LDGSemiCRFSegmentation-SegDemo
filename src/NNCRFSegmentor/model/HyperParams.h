#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	// must assign
	int wordcontext;
	int hiddensize;
	int rnnhiddensize;
	dtype dropOut;


	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization



	//auto generated
	int wordwindow;
	int wordDim;
	vector<int> typeDims;
	int unitsize;
	int inputsize;
	int labelSize;
	int batch;
public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		wordcontext = opt.wordcontext;
		hiddensize = opt.hiddenSize;
		rnnhiddensize = opt.rnnHiddenSize;
		dropOut = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		batch = opt.batchSize;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){
		cout << "======HyperParam======" << endl;
		cout << "wordcontext = " << wordcontext << endl;
		cout << "hiddensize = " << hiddensize << endl;
		cout << "rnnhiddensize = " << rnnhiddensize << endl;
		cout << "dropOut = " << dropOut << endl;


		cout << "nnRegular = " << nnRegular << endl; 
		cout << "adaAlpha = "<< adaAlpha << endl; 
		cout << "adaEps = " << adaEps << endl; 

		cout << "wordwindow = "<< wordwindow << endl;
		cout << "wordDim = " << wordDim << endl;
		int typeSize = typeDims.size();
		cout << typeSize << endl;
		for (int idx = 0; idx < typeSize; idx++)
			cout << "typeDims[" << idx << "] = " << typeDims[idx] << endl;
		cout << "unitsize = " << unitsize << endl;
		cout << "inputsize = " << inputsize << endl;
		cout << "labelSize = " << labelSize << endl;
		cout << "batch = " << batch << endl;
		cout << "======HyperParam======" << endl;
	}

	void saveModel(std::ofstream& os)const {
		os << wordcontext << endl;
		os << hiddensize << endl;
		os << rnnhiddensize << endl;
		os << dropOut << endl;


		os << nnRegular << endl; 
		os << adaAlpha << endl; 
		os << adaEps << endl; 

		os << wordwindow << endl;
		os << wordDim << endl;
		int typeSize = typeDims.size();
		os << typeSize << endl;
		for (int idx = 0; idx < typeSize; idx++)
		 os << typeDims[idx] << endl;
		os << unitsize << endl;
		os << inputsize << endl;
		os << labelSize << endl;
		os << batch << endl;
	}

	void loadModel(std::ifstream& is) {
		is >> wordcontext;
		is >> hiddensize;
		is >> rnnhiddensize;
		is >> dropOut;

		is >> nnRegular; 
		is >> adaAlpha; 
		is >> adaEps; 

		is >> wordwindow;
		is >> wordDim;
		int typeSize;
		is >> typeSize;
		typeDims.resize(typeSize);
		for (int idx = 0; idx < typeSize; idx++)
		 is >> typeDims[idx];
		is >> unitsize;
		is >> inputsize;
		is >> labelSize;
		is >> batch;
	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */