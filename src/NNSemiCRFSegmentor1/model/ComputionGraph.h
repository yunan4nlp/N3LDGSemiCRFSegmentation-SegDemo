#ifndef SRC_GraphBuilder_H_
#define SRC_GraphBuilder_H_

#include "ModelParams.h"
#include "Segmentation.h"

struct GraphBuilder {
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	int max_seg_length;
	int type_num;

	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;


	BucketNode bucket_node_left;
	BucketNode bucket_node;
	vector<PSubNode> sub_left;
	vector<PSubNode> sub_right;
	vector<BiNode> sub_hidden;

	vector<LinearNode> output;

	NRMat<PNode> poutput; //use to store pointer matrix of outputs
	Graph *_pcg;

public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int maxsegLen, int typeNum){
		max_seg_length = maxsegLen;
		type_num = typeNum;
		int segNum = sent_length * max_seg_length;
		resizeVec(word_inputs, sent_length, type_num + 1);		
		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		sub_left.resize(segNum);
		sub_right.resize(segNum);
		sub_hidden.resize(segNum);
		output.resize(segNum);
	}

	inline void clear(){
		clearVec(word_inputs);
		token_repsents.clear();
		word_window.clear();
		word_hidden1.clear();
		left_lstm.clear();
		right_lstm.clear();
		sub_left.clear();
		sub_right.clear();
		sub_hidden.clear();
		output.clear();	
	}


public:
	inline void initial(Graph* _pcg, ModelParams& model_params, HyperParams& hyper_params){
		this->_pcg = _pcg;
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model_params._words);
			word_inputs[idx][0].init(hyper_params.wordDim, hyper_params.dropProb);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model_params._types[idy - 1]);
				word_inputs[idx][idy].init(hyper_params.wordDim, hyper_params.dropProb);
			}

			token_repsents[idx].init(hyper_params.unitSize, -1);
			word_hidden1[idx].setParam(&model_params._tanh1_project);
			word_hidden1[idx].init(hyper_params.hiddenSize1, hyper_params.dropProb);
		}	
		word_window.init(hyper_params.unitSize, hyper_params.wordContext);
		left_lstm.init(&model_params._left_lstm_project, hyper_params.dropProb, true);
		right_lstm.init(&model_params._right_lstm_project, hyper_params.dropProb, false);
		bucket_node.init(hyper_params.rnnHiddenSize, -1);
		for(int idx = 0; idx < output.size(); idx++){
			sub_left[idx].init(hyper_params.rnnHiddenSize, -1);
			sub_right[idx].init(hyper_params.rnnHiddenSize, -1);
			sub_hidden[idx].setParam(&model_params._lstm_combine);
			sub_hidden[idx].init(hyper_params.segHiddenSize, hyper_params.dropProb);

			output[idx].setParam(&model_params._olayer_linear);
			output[idx].init(hyper_params.segLabelSize, -1);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear nodes

		_pcg->train = bTrain;
		//second step: build 
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(_pcg, feature.words[0]);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(_pcg, feature.types[idy - 1]);
			}

			token_repsents[idx].forward(_pcg, getPNodes(word_inputs[idx], word_inputs[idx].size()));
		}

		//windowlized
		word_window.forward(_pcg, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(_pcg, &(word_window._outputs[idx]));

		}

		left_lstm.forward(_pcg, getPNodes(word_hidden1, seq_size));
		right_lstm.forward(_pcg, getPNodes(word_hidden1, seq_size));


		static int offset;
		vector<PNode> segnodes;
		bucket_node.forward(_pcg, 0);
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			segnodes.clear();
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				//cout << offset << ", " << dist << endl;
				int i = idx - 1;
				int j = idx + dist;
				int count = offset + dist;
				if(i < 0) { 
					sub_left[count].forward(_pcg, &left_lstm._hiddens[j], &bucket_node);
				}
				else
					sub_left[count].forward(_pcg, &left_lstm._hiddens[j], &left_lstm._hiddens[i]);
				i = idx;
				j = idx + dist + 1;
				if (j >= seq_size) {
					sub_right[count].forward(_pcg, &right_lstm._hiddens[i], &bucket_node);
				}
				else
					sub_right[count].forward(_pcg, &right_lstm._hiddens[i], &right_lstm._hiddens[j]);
				sub_hidden[count].forward(_pcg, &sub_left[count], &sub_right[count]);
			}
		}
		
		poutput.resize(seq_size, max_seg_length);
		poutput = NULL;
		offset = 0;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				output[offset + dist].forward(_pcg, &(sub_hidden[offset + dist]));
				poutput[idx][dist] = &output[offset + dist];
			}
		}
	}

};

#endif /* SRC_GraphBuilder_H_ */

