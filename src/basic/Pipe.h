#ifndef _JST_PIPE_
#define _JST_PIPE_

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include "Instance.h"
#include "InstanceReader.h"
#include "InstanceWriter.h"
#include <iterator>

using namespace std;

//#define MAX_BUFFER_SIZE 256

class Pipe {
public:
  Pipe() {
    m_jstReader = new InstanceReader();
    m_jstWriter = new InstanceWriter();
	max_sentense_size = 1024;
  }

  ~Pipe(void) {
    if (m_jstReader)
      delete m_jstReader;
    if (m_jstWriter)
      delete m_jstWriter;
  }

  int initInputFile(const char *filename) {
    if (0 != m_jstReader->startReading(filename))
      return -1;
    return 0;
  }

  void uninitInputFile() {
    if (m_jstWriter)
      m_jstReader->finishReading();
  }

  int initOutputFile(const char *filename) {
    if (0 != m_jstWriter->startWriting(filename))
      return -1;
    return 0;
  }

  void uninitOutputFile() {
    if (m_jstWriter)
      m_jstWriter->finishWriting();
  }

  int outputAllInstances(const string& m_strOutFile, const vector<Instance>& vecInstances, bool bFile = true) {

    initOutputFile(m_strOutFile.c_str());
    static int instNum;
    instNum = vecInstances.size();
    for (int idx = 0; idx < instNum; idx++) {
      if (0 != m_jstWriter->write(&(vecInstances[idx]), bFile))
        return -1;
    }

    uninitOutputFile();
    return 0;
  }

  int outputSingleInstance(const Instance& inst, bool bFile = true) {
    if (0 != m_jstWriter->write(&inst, bFile))
      return -1;
    return 0;
  }

  Instance* nextInstance(bool bFile = true) {
    Instance *pInstance = m_jstReader->getNext(bFile);
    if (!pInstance || pInstance->labels.empty())
      return 0;

    return pInstance;
  }

  void readInstances(const string& m_strInFile, vector<Instance>& vecInstances, int maxInstance = -1, bool bFile = true) {
    vecInstances.clear();
    initInputFile(m_strInFile.c_str());

    Instance *pInstance = nextInstance(bFile);
    int numInstance = 0;

    while (pInstance) {
			if (pInstance->size() < max_sentense_size) {
				Instance trainInstance;
				trainInstance.copyValuesFrom(*pInstance);
				vecInstances.push_back(trainInstance);
				numInstance++;

				if (numInstance == maxInstance) {
					break;
				}
			}

			if (!bFile)
				break;
			else {
				pInstance = nextInstance(bFile);
			}
		}

    uninitInputFile();
		if (bFile) {
			cout << endl;
			cout << "instance num: " << numInstance << endl;
		}
  }

public:
	int max_sentense_size;

protected:
  Reader *m_jstReader;
  Writer *m_jstWriter;

};

#endif
