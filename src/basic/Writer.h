#ifndef _JST_WRITER_
#define _JST_WRITER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "Instance.h"

class Writer
{
public:
	Writer()
	{
		m_outf = &cout;
	}
	virtual ~Writer()
	{
		if (m_outf != &cout) {
			if (((ofstream *)m_outf)->is_open()) {
				((ofstream *)m_outf)->close();
				delete m_outf;
			}
		}
	}

	inline int startWriting(const string &filename) {
		if (filename != "") {
			m_outf = new std::ofstream(filename.c_str());

			if (!m_outf) {
				std::cout << "Writer::startWriting() open file err: " << filename << std::endl;
				return -1;
			}

		}
		return 0;
	}

	inline void finishWriting() {
		if (m_outf != &std::cout) {
			((std::ofstream *) m_outf)->close();
				delete m_outf;
		}
	}

	virtual int write(const Instance *pInstance, bool bFile) = 0;
protected:
	ostream *m_outf;
};

#endif

