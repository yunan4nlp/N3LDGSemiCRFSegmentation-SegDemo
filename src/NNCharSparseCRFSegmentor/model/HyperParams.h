#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	// must assign
	int charcontext;
	int charhiddensize;
	int wordcontext;
	int hiddensize;
	int rnnhiddensize;
	dtype dropOut;


	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization



	//auto generated
	int charwindow;
	int charDim;
	int charwindowoutput;

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
		charcontext = opt.charcontext;
		charhiddensize = opt.charhiddenSize;
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

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */