// Copyright 2008 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include "common.h"
#include <sstream>

char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";

namespace learning_lda {

bool IsValidProbDistribution(const TopicProbDistribution& dist) {
  const double kUnificationError = 0.00001;
  double sum_distribution = 0;
  for (int k = 0; k < dist.size(); ++k) {
    sum_distribution += dist[k];
  }
  return (sum_distribution - 1) * (sum_distribution - 1)
      <= kUnificationError;
}

int GetAccumulativeSample(const vector<double>& distribution) {
  double distribution_sum = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    distribution_sum += distribution[i];
  }

  double choice = RandDouble() * distribution_sum;
  double sum_so_far = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    sum_so_far += distribution[i];
    if (sum_so_far >= choice) {
      return i;
    }
  }

  LOG(FATAL) << "Failed to choose element from distribution of size "
             << distribution.size() << " and sum " << distribution_sum;

  return -1;
}

int LoadWordIndex(istream& in, map<string, int>& word_index_map) {
  string line;
  int maxindex = 0;
  while (getline(in, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
      line[0] != '\r' &&      // Skip empty lines.
      line[0] != '\n' &&      // Skip empty lines.
      line[0] != '$' &&      // Skip empty lines.
      line[0] != '#') {       // Skip comment lines.
        istringstream ss(line);
        int index;
        string word;
        ss >> index >> word;
        word_index_map[word] = index;
        if (maxindex < index)
          maxindex = index;
    }
  }
  return maxindex;
}


int LoadWordSet(istream& in ,map<string, int>& word_index_map ,set<int>& word_set) {
  string line;
  int sum = 0;
  while (getline(in, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
      line[0] != '\r' &&      // Skip empty lines.
      line[0] != '\n' &&      // Skip empty lines.
      line[0] != '#') {       // Skip comment lines.
        istringstream ss(line);
        string word;
        ss >> word;
        if (word_index_map.end() != word_index_map.find(word))
        {
          word_set.insert(word_index_map[word]);
          sum++;
        }
    }
  }
  return sum;
}

int LoadWordLex(map<string, int>& word_index_map,vector<string>& index_word_map){
  for (map<string, int>::const_iterator iter = word_index_map.begin();
    iter != word_index_map.end(); ++iter) {
      index_word_map[iter->second] = iter->first;
  }
  return index_word_map.size();
}


string generate_model_name(string path , int myid , int iter) {

  char buff[BUFF_SIZE_SHORT];

  if (0 <= iter) {
    sprintf(buff, "%d-%06d",myid, iter);
  }else{
    sprintf(buff, "%d-final",myid);
  }

  path += buff;
  path +=".txt";

  return path;
}


std::ostream& operator << (std::ostream& out, vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i] << " ";
  }
  return out;
}

}  // namespace learning_lda
