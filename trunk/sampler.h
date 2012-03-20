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

#ifndef _OPENSOURCE_GLDA_SAMPLER_H__
#define _OPENSOURCE_GLDA_SAMPLER_H__

#include "common.h"
#include "document.h"
#include "model.h"
#include "accumulative_model.h"
#include <sstream>

typedef map<int,std::pair< vector<int> ,map<int,vector<int> > > > RuleVec;
typedef map<int,std::pair< set<int> ,map<int,set<int> > > >  RuleSet;

namespace learning_lda {

// LDASampler trains LDA models and computes statistics about documents in
// LDA models.
  class LDATopicSplitRule{
    RuleVec split_rule_;
  public:
    bool LoadSplitRule(istream& in,const map<string, int>& word_index_map)
    {
      string line;

      RuleSet loc_split_rule_;
      while (getline(in, line)) {  // Each line is a training document.
        if (line.size() > 0 &&      // Skip empty lines.
          line[0] != '\r' &&      // Skip empty lines.
          line[0] != '\n' &&      // Skip empty lines.
          line[0] != '$' &&      // Skip empty lines.
          line[0] != '#') {       // Skip comment lines.
            std::istringstream ss(line);
            string word;
            ss >> word;
            if ( "Topic" != word)
              continue;
            int orgtopic;
            int curtopic;
            ss >> orgtopic;
            ss >> word;
            if ( "->" != word)
              continue;
            ss >> curtopic;

            loc_split_rule_.find(orgtopic);
            if (loc_split_rule_.end() == loc_split_rule_.find(orgtopic))
            {
                std::pair< set<int> ,map<int,set<int> > >  rule_;
                loc_split_rule_.insert(make_pair(orgtopic,rule_));
            }
            std::pair< set<int> ,map<int,set<int> > >  & currule =loc_split_rule_.find(orgtopic)->second;

            currule.first.insert(curtopic);
            map<int,set<int> >& word2topic = currule.second;


            while (ss >> word) {
              if (word_index_map.end() == word_index_map.find(word))
                continue;
              int wordid = word_index_map.find(word)->second;
              if (word2topic.end() == word2topic.find(wordid))
              {
                set<int> topiclist;
                word2topic.insert(make_pair(wordid,topiclist));
              }
              word2topic.find(wordid)->second.insert(curtopic);
            }
          }
      }

      map<int,std::pair< set<int> ,map<int,set<int> > > >::iterator iter;
      for (iter = loc_split_rule_.begin() ; loc_split_rule_.end() != iter ; iter++)
      {
        iter->first;
        set<int>::iterator iter2;
        vector<int> topicvec;
        for (iter2 = iter->second.first.begin(); iter->second.first.end() != iter2 ;iter2++)
        {
          topicvec.push_back(*iter2);
        }
        map<int,set<int> > & word2topic = iter->second.second;
        map<int,vector<int> >  word2topicvec;
        map<int,set<int> >::iterator iter3;
        for (iter3 = word2topic.begin();word2topic.end() != iter3 ; iter3++)
        {
          vector<int> wtopicvec;
          set<int>::iterator iter4;
          for (iter4 = iter3->second.begin(); iter3->second.end() != iter4 ;iter4++)
          {
            wtopicvec.push_back(*iter4);
          }
          word2topicvec.insert(make_pair(iter3->first,wtopicvec));
        }
        split_rule_.insert(make_pair(iter->first,make_pair(topicvec,word2topicvec)));
      }
      return true;
    }

      bool FindTopicList(int topicid,int wordid,vector<int> & topiclist)
      {
        RuleVec::iterator iter = split_rule_.find(topicid);
        if (split_rule_.end() == iter)
        {
          return false;
        }
        vector<int> & default_list = iter->second.first;
        map<int,vector<int> > & word2topiclist = iter->second.second;
        if (word2topiclist.end() == word2topiclist.find(wordid))
        {
          topiclist = default_list;
        }
        else
        {
          topiclist = word2topiclist.find(wordid)->second;
        }
        return true;
      }
  };

class LDASampler {
 public:
  // alpha and beta are the Gibbs sampling symmetric hyperparameters.
  // model is the model to use.
  LDASampler(double alpha, double beta,
             LDAModel* model,
             LDAAccumulativeModel* accum_model);

  ~LDASampler() {}

  // Given a corpus, whose every document have been initialized (i.e.,
  // every word occurrences has a (randomly) assigned topic,
  // initialize model_ to count the word-topic co-occurrences.
  void InitModelGivenTopics(const LDACorpus& corpus);

  // Performs one round of Gibbs sampling on documents in the corpus
  // by invoking SampleNewTopicsForDocument(...).  If we are to train
  // a model given training data, we should set train_model to true,
  // and the algorithm updates model_ during Gibbs sampling.
  // Otherwise, if we are to sample the latent topics of a query
  // document, we should set train_model to false.  If train_model is
  // true, burn_in indicates should we accumulate the current estimate
  // to accum_model_.  For the first certain number of iterations,
  // where the algorithm has not converged yet, you should set burn_in
  // to false.  After that, we should set burn_in to true.
  void DoIteration(LDACorpus* corpus, bool train_model, bool burn_in);
  void DoIteration2(LDACorpus* corpus, bool train_model, bool burn_in);

  // Performs one round of Gibbs sampling on a document.  Updates
  // document's topic assignments.  For learning, update_model_=true,
  // for sampling topics of a query, update_model_==false.
  void SampleNewTopicsForDocument(LDADocument* document,
                                  bool update_model);

  // The core of the Gibbs sampling process.  Compute the full conditional
  // posterior distribution of topic assignments to the indicated word.
  //
  // That is, holding all word-topic assignments constant, except for the
  // indicated one, compute a non-normalized probability distribution over
  // topics for the indicated word occurrence.
  void GenerateTopicDistributionForWord(const LDADocument& document,
      int word, int current_word_topic, bool train_model,
      vector<double>* distribution) const;

  void SampleNewTopicsForDocument2(LDADocument* document,
    bool update_model);

  void GenerateTopicDistributionForWord2(const LDADocument& document,
    int word, int current_word_topic, bool train_model,
    const vector<int>& topic_list,vector<double>* distribution) const;

  void AdjustCorpusWithRule(LDACorpus& corpus,
    int new_topic_num,
    set<int> & new_words,
    LDATopicSplitRule &adjust_rule);
    

  // Computes the log likelihood of a document.
  double LogLikelihood(LDADocument* document) const;

 private:
  const double alpha_;
  const double beta_;
  LDAModel* model_;
  LDAAccumulativeModel* accum_model_;
};


}  // namespace learning_lda

#endif  // _OPENSOURCE_GLDA_SAMPLER_H__
