#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "Instance.h"

class Reader
{
public:
	Reader()
	{
		m_inf = &cin;
	}

	virtual ~Reader()
	{
		if (m_inf != &cin) {
			if (((ifstream *)m_inf)->is_open()) {
				((ifstream *)m_inf)->close();
				delete m_inf;
			}
		}
	}

	int startReading(const string &filename) {
		if(filename != "") {
			m_inf = new std::ifstream(filename.c_str());
			if (!m_inf) {
				cout << "Reader::startReading() open file err: " << filename << endl;
				return -1;
			}
		} 
		return 0;
	}

	void finishReading() {
		if(m_inf != &std::cin){
			if (((ifstream *)m_inf)->is_open())
				((ifstream *)m_inf)->close();
				delete m_inf;
				m_inf = &cin;
		}
	}

	virtual Instance *getNext(bool bFile) = 0;
protected:
	istream *m_inf;

	int m_numInstance;

	Instance m_instance;
};

#endif

