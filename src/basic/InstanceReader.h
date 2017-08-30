#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */

inline bool my_getline(istream &inf, string &line) {
	if (!getline(inf, line))
		return false;
	int end = line.size() - 1;
	while (end >= 0 && (line[end] == '\r' || line[end] == '\n')) {
		line.erase(end--);
	}

	return true;
}

class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext(bool bFile = true) {
		if (bFile)
		{
			m_instance.clear();
			vector<string> vecLine;
			while (1) {
				string strLine;
				if (!my_getline(*m_inf, strLine)) {
					break;
				}
				if (strLine.empty())
					break;
				vecLine.push_back(strLine);
			}

			int length = vecLine.size();

			m_instance.allocate(length);
			static string::size_type pos;

			for (int i = 0; i < length; ++i) {
				vector<string> vecInfo;
				split_bychar(vecLine[i], vecInfo, ' ');
				int veclength = vecInfo.size();
				m_instance.labels[i] = vecInfo[veclength - 1];
				//m_instance.words[i] = normalize_to_lowerwithdigit(vecInfo[0]);
				m_instance.words[i] = vecInfo[0];
				for (int j = 1; j < veclength - 1; j++) {
					if (is_startwith(vecInfo[j], "[S]"))
						m_instance.sparsefeatures[i].push_back(vecInfo[j]);
					if (is_startwith(vecInfo[j], "[c]"))
						m_instance.charfeatures[i].push_back(vecInfo[j].substr(pos + 3));
					if (is_startwith(vecInfo[j], "[T")) {
						pos = vecInfo[j].find_first_of("]");
						m_instance.typefeatures[i].push_back(vecInfo[j].substr(pos + 1));
					}
				}
			}
		}
		else {
			m_instance.clear();
			string strLine;
			if (!my_getline(*m_inf, strLine))
				return NULL;
			if (strLine.empty())
				return NULL;
			vector<string> words;
			getCharactersFromUTF8String(strLine, words);
			int word_num = words.size();
			m_instance.allocate(word_num);
			for (int idx = 0; idx < word_num; idx++) {
				m_instance.words[idx] = words[idx];
				if (idx == 0)
					m_instance.typefeatures[idx].push_back("<s>" + m_instance.words[idx]);
				else
					m_instance.typefeatures[idx].push_back(m_instance.words[idx - 1] + m_instance.words[idx]);
				m_instance.labels[idx] = unknownkey;
			}
		}

		return &m_instance;
	}
};

#endif

