#ifndef _RICH_RANGE_AGENT_HPP_
#define _RICH_RANGE_AGENT_HPP_

#ifdef RICH_MPI

#include "utils/queryAgent/QueryAgent.hpp"
#include "utils/queryAgent/DuplicationRemover.hpp"
#include "3D/range/finders/RangeFinder.hpp"
#include "3D/environment/EnvironmentAgent.h"

struct RangeQueryData
{
    _3DPoint center;
    typename _3DPoint::coord_type radius;
};


/**
 * The range agent is responsible for running batches of range queries. A batch is a collection of queries, and a range query is an instance of the `RangeQueryData` class, containing a point and a requested radius.
 * The range agent switches between roles - sending queries, receiving answers, and answering for incoming queries. It also supports duplications removal, and returns the results rearranged by processes (what are the points that were received from each one, and what points I sent to each one).
 * In order to answer for incoming requests, a range finder is required. A range finder is an object which holds a list of points, and can answer for range queries.
*/
class RangeAgent
{
private:
    class RangeAnswerAgent : public AnswerAgent<RangeQueryData, IndexedVector3D>
    {
    public:
        RangeAnswerAgent(const RangeFinder *rangeFinder): rangeFinder(rangeFinder){};

        std::vector<IndexedVector3D> answer(const RangeQueryData &query, int _rank) override
        {
            return this->rangeFinder->range(query.center, query.radius);
        }

    private:
        const RangeFinder *rangeFinder;
    };

    class RangeTalkAgent : public TalkAgent<RangeQueryData>
    {
    public:
        RangeTalkAgent(const EnvironmentAgent *envAgent): envAgent(envAgent){};

        inline _set<int> getTalkList(const RangeQueryData &query) const override
        {
            return this->envAgent->getIntersectingRanks(Vector3D(query.center.x, query.center.y, query.center.z), query.radius);
        }

    private:
        const EnvironmentAgent *envAgent;
    };

public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    RangeAgent(const EnvironmentAgent *envAgent, RangeFinder *rangeFinder, const MPI_Comm &comm = MPI_COMM_WORLD)
    {
        this->regularAnsAgent = new RangeAnswerAgent(rangeFinder);
        this->ansAgent = new DuplicationRemover<RangeQueryData, _3DPoint, IndexedVector3D>(this->regularAnsAgent);
        this->talkAgent = new RangeTalkAgent(envAgent);
        this->queryAgent = new QueryAgent<RangeQueryData, _3DPoint>(this->talkAgent, this->ansAgent, comm);
    }

    ~RangeAgent()
    {
        delete this->queryAgent;
        delete this->talkAgent;
        delete this->ansAgent;
        delete this->regularAnsAgent;
    }

    inline QueryBatchInfo<RangeQueryData, _3DPoint> runBatch( std::queue<RangeQueryData> &queries)
    {
        return this->queryAgent->runBatch(queries);
    };

    inline std::vector<std::vector<std::size_t>> &getSentPoints(){return this->queryAgent->getSentData();};
    inline std::vector<std::vector<std::size_t>> &getRecvPoints(){return this->queryAgent->getRecvData();};
    inline std::vector<int> &getSentProc(){return this->queryAgent->getSentProc();};
    inline std::vector<int> &getRecvProc(){return this->queryAgent->getRecvProc();};

private:
    RangeAnswerAgent *regularAnsAgent;
    AnswerAgent<RangeQueryData, _3DPoint> *ansAgent;
    QueryAgent<RangeQueryData, _3DPoint> *queryAgent;
    TalkAgent<RangeQueryData> *talkAgent;
};

#endif // RICH_MPI

#endif // _RICH_RANGE_AGENT_HPP_