#if 0
#ifndef _DUPLICATOR_REMOVER_HPP
#define _DUPLICATOR_REMOVER_HPP

#include <boost/container/flat_set.hpp>

/**
 * Decorator for answer agent (a type of agent that removed duplications)
*/
template<typename QueryData, typename AnswerType, typename IndexedAnswerType>
class DuplicationRemover : public AnswerAgent<QueryData, AnswerType>
{
    template<typename T>
    using _set = boost::container::flat_set<T>;

public:
    DuplicationRemover(AnswerAgent<QueryData, IndexedAnswerType> *agent): agent(agent){};
    
    std::vector<AnswerType> answer(const QueryData &query, int _rank) override
    {
        std::vector<int> &sentProc = this->agent->getSentProc();
        std::vector<std::vector<size_t>> &sentData = this->agent->getSentData();

        std::vector<IndexedAnswerType> unfilteredResult = this->agent->answer(query, _rank);
        std::vector<AnswerType> result;

        if(unfilteredResult.empty())
        {
            return result;
        }
        size_t rankIdx = std::distance(sentProc.begin(), std::find(sentProc.begin(), sentProc.end(), _rank));
        if(rankIdx == sentProc.size())
        {
            // `_rank` is new
            sentProc.push_back(_rank);
            sentData.emplace_back(std::vector<size_t>());
            this->sentDataSet.emplace_back(_set<size_t>());
        }
        RangeFinder::_set<size_t> &_rankSet = this->sentDataSet[rankIdx];

        for(const IndexedAnswerType &_data : unfilteredResult)
        {
            size_t idx = _data.getIndex();
            AnswerType data = _data.getData();

            if(_rankSet.find(idx) == _rankSet.end())
            {
                // `_data` was not sent before
                result.push_back(data);
                _rankSet.insert(idx);
                sentData[rankIdx].push_back(idx);
            }
        }
        return result;
    }

private:
    AnswerAgent<QueryData, IndexedAnswerType> *agent;
    std::vector<RangeFinder::_set<size_t>> sentDataSet;
};

#endif // _DUPLICATOR_REMOVER_HPP
#endif // 0