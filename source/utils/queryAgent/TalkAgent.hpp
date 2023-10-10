#ifndef _TALK_AGENT_HPP
#define _TALK_AGENT_HPP

template<typename QueryData>
class TalkAgent
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    virtual _set<int> getTalkList(const QueryData &query) const = 0;
};

#endif // _TALK_AGENT_HPP