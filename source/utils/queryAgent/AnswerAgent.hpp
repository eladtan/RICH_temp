#ifndef _ANSWER_AGENT_HPP
#define _ANSWER_AGENT_HPP

#include <iostream> // todo remove
#include <vector>

template<typename QueryData, typename AnswerType>
class AnswerAgent
{
public:
    virtual ~AnswerAgent() = default;
    
    virtual std::vector<AnswerType> answer(const QueryData &query, int _rank) = 0;

    virtual std::vector<int> &getSentProc()
    {
        return this->sentProc;
    }

    virtual std::vector<std::vector<size_t>> &getSentData()
    {
        return this->sentData;
    }

protected:
    std::vector<int> sentProc;
    std::vector<std::vector<size_t>> sentData;
};

#endif // _ANSWER_AGENT_HPP