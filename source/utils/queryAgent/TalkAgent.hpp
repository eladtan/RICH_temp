#ifndef _TALK_AGENT_HPP
#define _TALK_AGENT_HPP

#ifdef RICH_MPI

#include <boost/container/flat_set.hpp>

template<typename QueryData>
class TalkAgent
{
public:
    using RanksSet = boost::container::flat_set<int>;

    virtual ~TalkAgent() = default;

    virtual RanksSet getTalkList(const QueryData &query) const = 0;
};

#endif // RICH_MPI

#endif // _TALK_AGENT_HPP