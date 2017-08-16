#ifndef SRC_GraphBuilder_H_
#define SRC_GraphBuilder_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder {
public:
	const static int max_sentence_length = 1024;
	const static int max_char_length = 16;

public:
	// node instances
	vector<vector<LookupNode> > char_inputs;
	vector<WindowBuilder> char_windows;
	vector<vector<UniNode> > char_hiddens;
	vector<MaxPoolNode> char_max_poolings;
	vector<MinPoolNode> char_min_poolings;
	vector<AvgPoolNode> char_avg_poolings;
	vector<ConcatNode> char_concat;

	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;
	vector<ConcatNode> word_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<BiNode> word_hidden2;
	vector<UniNode> word_hidden3;
	vector<SparseNode> sparse_feats;
	vector<PAddNode> ns_combines;
	vector<LinearNode> output;


	int type_num;


	Graph *_pcg;
	// node pointers
public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int typeNum, int char_length){
		type_num = typeNum;
		resizeVec(word_inputs, sent_length, type_num + 1);
		token_repsents.resize(sent_length);
		word_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		word_hidden3.resize(sent_length);
		sparse_feats.resize(sent_length);
		ns_combines.resize(sent_length);
		output.resize(sent_length);

		char_inputs.resize(sent_length);
		char_windows.resize(sent_length);
		char_hiddens.resize(sent_length);
		char_max_poolings.resize(sent_length);
		char_min_poolings.resize(sent_length);
		char_avg_poolings.resize(sent_length);
		for (int idx = 0; idx < sent_length; idx++) {
			char_inputs[idx].resize(char_length);
			char_windows[idx].resize(char_length);
			char_hiddens[idx].resize(char_length);
			char_max_poolings[idx].setParam(char_length);
			char_min_poolings[idx].setParam(char_length);
			char_avg_poolings[idx].setParam(char_length);
		}
		char_concat.resize(sent_length);
	}

	inline void clear(){
		//Graph::clear();
		clearVec(word_inputs);
		clearVec(char_inputs);
		char_hiddens.clear();
		char_max_poolings.clear();
		char_min_poolings.clear();
		char_avg_poolings.clear();
		char_concat.clear();

		token_repsents.clear();
		word_repsents.clear();
		word_window.clear();
		word_hidden1.clear();
		left_lstm.clear();
		right_lstm.clear();
		word_hidden2.clear();
		word_hidden3.clear();
		sparse_feats.clear();
		ns_combines.clear();
		output.clear();
	}

public:
	inline void initial(Graph* _pcg, ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem){
		this->_pcg = _pcg;

		int maxsize = word_inputs.size();
		for (int idx = 0; idx < maxsize; idx++) {
			word_inputs[idx][0].setParam(&model.words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model.types[idy - 1]);
			}

			int maxcharsize = char_inputs[idx].size();
			for (int idy = 0; idy < maxcharsize; idy++) {
				char_inputs[idx][idy].setParam(&model.chars);
				char_inputs[idx][idy].init(opts.charDim, opts.dropOut, mem);
				char_hiddens[idx][idy].setParam(&model.char_tanh_project);
				char_hiddens[idx][idy].init(opts.charhiddensize, opts.dropOut, mem);
			}
			char_windows[idx].init(opts.charDim, opts.charcontext, mem);
			char_max_poolings[idx].init(opts.charhiddensize, -1, mem);
			char_min_poolings[idx].init(opts.charhiddensize, -1, mem);
			char_avg_poolings[idx].init(opts.charhiddensize, -1, mem);
			char_concat[idx].init(opts.charhiddensize * 3, -1, mem);

			word_hidden1[idx].setParam(&model.tanh1_project);
			word_hidden2[idx].setParam(&model.tanh2_project);
			word_hidden3[idx].setParam(&model.tanh3_project);
			sparse_feats[idx].setParam(&model.sparse_params);
			output[idx].setParam(&model.olayer_linear);
		}
		
		word_window.init(opts.inputsize, opts.wordcontext, mem);
		left_lstm.init(&model.left_lstm_project, opts.dropOut, true, mem);
		right_lstm.init(&model.right_lstm_project, opts.dropOut, false, mem);

		for (int idx = 0; idx < maxsize; idx++){
			word_inputs[idx][0].init(opts.wordDim, opts.dropOut, mem);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].init(opts.typeDims[idy-1], opts.dropOut, mem);
			}
			token_repsents[idx].init(opts.unitsize, -1, mem);
			word_repsents[idx].init(opts.inputsize, -1, mem);
			word_hidden1[idx].init(opts.hiddensize, opts.dropOut, mem);
			word_hidden2[idx].init(opts.hiddensize, opts.dropOut, mem);
			word_hidden3[idx].init(opts.hiddensize, -1, mem);
			sparse_feats[idx].init(opts.hiddensize, -1, mem);
			ns_combines[idx].init(opts.hiddensize, -1, mem);
			output[idx].init(opts.labelSize, -1, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear value
//		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		_pcg->train = bTrain;
		// second step: build graph
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
			
			int char_size = feature.chars.size() > max_char_length ? max_char_length : feature.chars.size();
			for (int idy = 0; idy < char_size; idy++)
				char_inputs[idx][idy].forward(_pcg, feature.chars[idy]);
			char_windows[idx].forward(_pcg, getPNodes(char_inputs[idx], char_size));
			for (int idy = 0; idy < char_size; idy++)
				char_hiddens[idx][idy].forward(_pcg, &char_windows[idx]._outputs[idy]);
			char_max_poolings[idx].forward(_pcg, getPNodes(char_hiddens[idx], char_size));
			char_min_poolings[idx].forward(_pcg, getPNodes(char_hiddens[idx], char_size));
			char_avg_poolings[idx].forward(_pcg, getPNodes(char_hiddens[idx], char_size));
			char_concat[idx].forward(_pcg, &char_avg_poolings[idx], &char_max_poolings[idx], &char_min_poolings[idx]);

			token_repsents[idx].forward(_pcg, getPNodes(word_inputs[idx], word_inputs[idx].size()));
			word_repsents[idx].forward(_pcg, &token_repsents[idx], &char_concat[idx]);
		}

		//windowlized
		word_window.forward(_pcg, getPNodes(word_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(_pcg, &(word_window._outputs[idx]));
		}

		left_lstm.forward(_pcg, getPNodes(word_hidden1, seq_size));
		right_lstm.forward(_pcg, getPNodes(word_hidden1, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			const Feature& feature = features[idx];
			word_hidden2[idx].forward(_pcg, &(left_lstm._hiddens[idx]), &(right_lstm._hiddens[idx]));
			word_hidden3[idx].forward(_pcg, &(word_hidden2[idx]));
			sparse_feats[idx].forward(_pcg, feature.linear_features);
			ns_combines[idx].forward(_pcg, &sparse_feats[idx], &word_hidden3[idx]);
			output[idx].forward(_pcg, &(ns_combines[idx]));
		}
	}

};

#endif /* SRC_GraphBuilder_H_ */