#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"
#include "CRFMLLoss.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	LSTM1Params lstm_params;
	UniParams olayer_linear;

public:
	Alphabet labelAlpha; // should be initialized outside
	//CRFMLLoss loss;
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.labelSize = labelAlpha.size();
		opts.wordwindow = opts.wordcontext * 2 + 1;
		
		lstm_params.initial(opts.rnnhiddensize, opts.wordDim, mem);
		olayer_linear.initial(opts.labelSize, opts.rnnhiddensize, false, mem);

		//loss.initial(opts.labelSize);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		lstm_params.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
		//loss.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		//checkgrad.add(&(loss.T), "loss.T");

		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");

		checkgrad.add(&(lstm_params.input.W1), "lstm_params.input.W1");
		checkgrad.add(&(lstm_params.input.W2), "lstm_params.input.W2");
		checkgrad.add(&(lstm_params.input.b), "lstm_params.input.b");

		checkgrad.add(&(lstm_params.cell.W1), "lstm_params.cell.W1");
		checkgrad.add(&(lstm_params.cell.W2), "lstm_params.cell.W2");
		checkgrad.add(&(lstm_params.cell.b), "lstm_params.cell.b");

		checkgrad.add(&(words.E), "_words.E");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */