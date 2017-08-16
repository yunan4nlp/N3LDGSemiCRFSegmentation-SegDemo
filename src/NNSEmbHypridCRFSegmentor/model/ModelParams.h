#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "BMESSegmentation.h"
#include "HyperParams.h"
#include "CRFMLLoss.h"
#include "Semi0CRFMLLoss.h"

class ModelParams{
public:
	LSTM1Params _left_lstm_project; //left lstm
	LSTM1Params _right_lstm_project; //right lstm
	UniParams _tanh1_project; // hidden
	BiParams _tanh2_project; // hidden
	SegParams _seglayer_project; //segmentation
	BiParams _segtanh_project; //
	UniParams _olayer_linear; // output
	UniParams _olayerbmes_linear; // output


	//SoftMaxLoss _loss;
	Semi0CRFMLLoss _loss;
	CRFMLLoss _bmesloss;

public: // follow paramsters shoulde be initialized outside
	vector<LookupTable> _types;
	vector<Alphabet> _type_alphas;
	Alphabet _label_alpha;
	Alphabet _seg_label_alpha;
	Alphabet _word_alpha;
	Alphabet _seg_alpha;
	LookupTable _words;
	LookupTable _segs;

public:
	bool initial(HyperParams& hyper_params){
		if (_words.nVSize <= 0 || _label_alpha.size() < 0 || _loss.labelSize <= 0 || _loss.maxLen <=0  || _segs.nVSize <=0){
			std::cout << "Please initialize embeddings before this." << std::endl;
			return false;
		}
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.wordDim = _words.nDim;
		hyper_params.segDim = _segs.nDim;
		hyper_params.unitSize = hyper_params.wordDim;
		hyper_params.typeDims.clear();
		for (int idx = 0; idx < _types.size(); idx++)
		{
			if (_types[idx].nVSize <= 0 || _type_alphas[idx].size() <= 0)
				return false;
			hyper_params.typeDims.push_back(_types[idx].nDim);
			hyper_params.unitSize += hyper_params.typeDims[idx];
		}
		hyper_params.segLabelSize = _seg_label_alpha.size();
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.inputSize = hyper_params.wordWindow * hyper_params.unitSize;

		_tanh1_project.initial(hyper_params.hiddenSize1, hyper_params.inputSize ,true);
		_left_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_right_lstm_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize1);
		_tanh2_project.initial(hyper_params.hiddenSize2, hyper_params.rnnHiddenSize, hyper_params.rnnHiddenSize, true);
		_seglayer_project.initial(hyper_params.segHiddenSize, hyper_params.hiddenSize2, hyper_params.hiddenSize1);
		_segtanh_project.initial(hyper_params.segHiddenSize, hyper_params.segHiddenSize, hyper_params.segDim, true);
		_olayer_linear.initial(hyper_params.segLabelSize, hyper_params.segHiddenSize, false);
		_olayerbmes_linear.initial(hyper_params.labelSize, hyper_params.hiddenSize1, false);
		_bmesloss.initial(hyper_params.labelSize);

		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		for (int idx = 0; idx < _types.size(); idx++)
			_types[idx].exportAdaParams(ada);
		_tanh1_project.exportAdaParams(ada);
		_left_lstm_project.exportAdaParams(ada);
		_right_lstm_project.exportAdaParams(ada);
		_tanh2_project.exportAdaParams(ada);
		_seglayer_project.exportAdaParams(ada);
		_segtanh_project.exportAdaParams(ada);
		_olayer_linear.exportAdaParams(ada);
		_olayerbmes_linear.exportAdaParams(ada);
		_bmesloss.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(_tanh1_project.W), "_tan1_project.W");
		checkgrad.add(&(_tanh1_project.b), "_tan1_project.b");

		checkgrad.add(&(_tanh2_project.W1), "_tan1_project.W1");
		checkgrad.add(&(_tanh2_project.W2), "_tan1_project.W2");
		checkgrad.add(&(_tanh2_project.b), "_tan2_project.b");
		checkgrad.add(&_seglayer_project.B.W, "_seglayer_project.B.W");
		checkgrad.add(&_seglayer_project.M.W, "_seglayer_project.M.W");
		checkgrad.add(&_seglayer_project.E.W, "_seglayer_project.E.W");
		checkgrad.add(&_seglayer_project.S.W, "_seglayer_project.S.W");
		checkgrad.add(&_segtanh_project.W1, "_segtanh_project.W1");
		checkgrad.add(&_segtanh_project.W2, "_segtanh_project.W2");
		checkgrad.add(&_segtanh_project.b, "_segtanh_project.b");
		checkgrad.add(&_olayerbmes_linear.W, "_olayerbmes_linear.W");
		checkgrad.add(&_bmesloss.T, "_bmesloss.T");
	}

	void saveModel(){
	}

	void loadModel(const string& infile){
	}
};

#endif /*SRC_ModelParams_H_*/
