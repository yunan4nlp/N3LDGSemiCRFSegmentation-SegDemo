#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"
#include "Segmentation.h"
#include "Semi0CRFMLLoss.h"

class ModelParams{
public:

	LSTM1Params _left_lstm_project; //left lstm
	LSTM1Params _right_lstm_project; //right lstm
	UniParams _tanh1_project; // hidden
	BiParams _lstm_combine;
	UniParams _olayer_linear; // output

	Semi0CRFMLLoss _loss;
	
public://follow parameters should be initialized outside
	vector<LookupTable> _types;
	vector<Alphabet> _type_alphas;
	LookupTable _words;
	Alphabet _word_alpha;
	Alphabet _label_alpha;
	Alphabet _seg_label_alpha;

public:
	bool initial(HyperParams& hyper_params){
		if(_words.nVSize <= 0 || _label_alpha.size() < 0)
			return false;
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.wordDim = _words.nDim;
		hyper_params.unitSize = hyper_params.wordDim;
		hyper_params.typeDims.clear();
		for(int idx = 0; idx < _types.size(); idx++)
		{
			if(_types[idx].nVSize <= 0 || _type_alphas[idx].size() <= 0)
				return false;
			hyper_params.typeDims.push_back(_types[idx].nDim);
			hyper_params.unitSize += hyper_params.typeDims[idx];
		}
		hyper_params.segLabelSize = _seg_label_alpha.size();
		hyper_params.inputSize = hyper_params.wordWindow * hyper_params.unitSize;

		_tanh1_project.initial(hyper_params.hiddenSize1, hyper_params.inputSize, true);
		_left_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_right_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_lstm_combine.initial(hyper_params.segHiddenSize, hyper_params.rnnHiddenSize, hyper_params.rnnHiddenSize, true);
		_olayer_linear.initial(hyper_params.segLabelSize, hyper_params.segHiddenSize, false);
		_loss.initial(hyper_params.maxLabelLength, hyper_params.maxsegLen);
		
		return true;
	}	

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		for(int idx = 0; idx < _types.size(); idx++)
			_types[idx].exportAdaParams(ada);
		_tanh1_project.exportAdaParams(ada);
		_left_lstm_project.exportAdaParams(ada);
		_right_lstm_project.exportAdaParams(ada);
		_lstm_combine.exportAdaParams(ada);
		_olayer_linear.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_words.E, "_words.E");
		checkgrad.add(&_tanh1_project.W, "_tanh1_project.W");
		checkgrad.add(&_tanh1_project.b, "_tanh1_project.b");
		checkgrad.add(&_olayer_linear.W, "_olayer_linear.W");
	}

	void saveModel(std::ofstream& os) const {
		 _left_lstm_project.save(os); 
		 _right_lstm_project.save(os);
		 _tanh1_project.save(os); 
		 _lstm_combine.save(os);
		 _olayer_linear.save(os);

		 _loss.save(os);

		 int typeAlphaSize = _type_alphas.size();
		 os << typeAlphaSize << endl;
		 for (int idx = 0; idx < typeAlphaSize; idx++)
			 _type_alphas[idx].write(os);
		 int typeSize = _types.size();
		 os << typeSize << endl;
		 for (int idx = 0; idx < typeSize; idx++)
			 _types[idx].save(os);

		 _word_alpha.write(os);
		 _words.save(os);

		 _seg_label_alpha.write(os);
		 _label_alpha.write(os);
	}

	void loadModel(std::ifstream& is){
		 _left_lstm_project.load(is); 
		 _right_lstm_project.load(is);
		 _tanh1_project.load(is); 
		 _lstm_combine.load(is);
		 _olayer_linear.load(is);

		 _loss.load(is);

		 int typeAlphaSize;
		 is >> typeAlphaSize;
		 _type_alphas.resize(typeAlphaSize);
		 for (int idx = 0; idx < typeAlphaSize; idx++)
			 _type_alphas[idx].read(is);
		 int typeSize;
		 is >> typeSize;
		 _types.resize(typeSize);
		 for (int idx = 0; idx < typeSize; idx++)
			 _types[idx].load(is, &_type_alphas[idx]);

		 _word_alpha.read(is);
		 _words.load(is, &_word_alpha);

		 _seg_label_alpha.read(is);
		 _label_alpha.read(is);
	}
};


#endif /*SRC_ModelParams_H_ */
