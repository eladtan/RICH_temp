#ifndef _RICH_RANGE_AGENT_HPP_
#define _RICH_RANGE_AGENT_HPP_

#ifdef RICH_MPI

#include "utils/queryAgent/QueryAgent.hpp"
#include "utils/queryAgent/DuplicationRemover.hpp"
#include "3D/range/finders/RangeFinder.hpp"
#include "3D/range/finders/utils/IndexedVector.hpp"
#include "3D/environment/EnvironmentAgent.h"
#include "3D/environment/DistributedOctEnvAgent.hpp"

#define ASK_ONLY_CLOSE 0
#define ASK_ALL 1

typedef struct RangeQueryData
{
    size_t pointIndex;
    _3DPoint center;
    _3DPoint extraPoint;
    coord_t radius;
    size_t maxPointsToGet;
    char whoToAsk;

    friend std::ostream &operator<<(std::ostream &stream, const RangeQueryData &query)
    {
        stream << "[center " << query.center << ", radius " << query.radius << ", extra point " << query.extraPoint << ", max2get " << query.maxPointsToGet << "]";
        return stream;
    }

} RangeQueryData;

#define NO_MAX_POINTS 1000000000

/**
 * The range agent is responsible for running batches of range queries. A batch is a collection of queries, and a range query is an instance of the `RangeQueryData` class, containing a point and a requested radius.
 * The range agent switches between roles - sending queries, receiving answers, and answering for incoming queries. It also supports duplications removal, and returns the results rearranged by processes (what are the points that were received from each one, and what points I sent to each one).
 * In order to answer for incoming requests, a range finder is required. A range finder is an object which holds a list of points, and can answer for range queries.
*/
class RangeAgent
{
private:
    class RangeAnswerAgent : public AnswerAgent<RangeQueryData, _3DPoint>
    {
        friend class RangeAgent;

    public:
        RangeAnswerAgent(const RangeFinder *rangeFinder): rangeFinder(rangeFinder){};

        std::vector<size_t> clearDuplication(const std::vector<size_t> &unfilteredResult, int _rank)
        {
            std::vector<int> &sentProc = this->sentProc;
            std::vector<std::vector<size_t>> &sentData = this->sentData;
            std::vector<RangeFinder::_set<size_t>> &sentDataSet = this->sentDataSet;
            std::vector<size_t> result;

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
                sentDataSet.emplace_back(RangeFinder::_set<size_t>());
            }
            RangeFinder::_set<size_t> &_rankSet = sentDataSet[rankIdx];

            for(const size_t &dataIdx : unfilteredResult)
            {
                if(_rankSet.find(dataIdx) == _rankSet.end())
                {
                    // `_data` was not sent before
                    result.push_back(dataIdx);
                    _rankSet.insert(dataIdx);
                    sentData[rankIdx].push_back(dataIdx);
                }
            }
            return result;
        }

        std::vector<_3DPoint> answer(const RangeQueryData &query, int _rank) override
        {
            std::vector<_3DPoint> result;
            std::vector<size_t> indicesResult;

            if(query.maxPointsToGet == 1)
            {
                size_t rankIndex = std::distance(this->sentProc.begin(), std::find(this->sentProc.begin(), this->sentProc.end(), _rank));
                const RangeFinder::_set<size_t> &ignore = (rankIndex == this->sentProc.size())? RangeFinder::_set<size_t>() : sentDataSet[rankIndex];
                indicesResult = this->rangeFinder->closestPointInSphere(Vector3D(query.center.x, query.center.y, query.center.z), query.radius, Vector3D(query.extraPoint.x, query.extraPoint.y, query.extraPoint.z), ignore);
            }
            else
            {
                indicesResult = this->rangeFinder->range(Vector3D(query.center.x, query.center.y, query.center.z), query.radius);
            }

            indicesResult = this->clearDuplication(indicesResult, _rank);

            result.reserve(indicesResult.size());
            for(const size_t &pointIdx : indicesResult)
            {
                result.push_back(_3DPoint(this->rangeFinder->getPoint(pointIdx)));
            }
            return result;
        }

    private:
        const RangeFinder *rangeFinder;
        std::vector<RangeFinder::_set<size_t>> sentDataSet;
    };

    class RangeTalkAgent : public TalkAgent<RangeQueryData>
    {
    public:
        template<typename K, typename V>
        using _map = boost::container::flat_map<K, V>;

        RangeTalkAgent(const EnvironmentAgent *envAgent, const MPI_Comm &comm = MPI_COMM_WORLD): envAgent(envAgent)
        {
            MPI_Comm_rank(comm, &this->rank);
            MPI_Comm_size(comm, &this->size);
        };

        inline EnvironmentAgent::RanksSet getTalkList(const RangeQueryData &query) const override
        {
            const DistributedOctEnvironmentAgent *distributedOctAgent = dynamic_cast<const DistributedOctEnvironmentAgent*>(this->envAgent);
            EnvironmentAgent::RanksSet intersectingRanks = this->envAgent->getIntersectingRanks(Vector3D(query.center.x, query.center.y, query.center.z), query.radius);
            if(intersectingRanks.empty())
            {
                std::cout << "query is " << query << std::endl;
                throw UniversalError("In range talk agent, should not reach here: the intersecting ranks list should at least contain the rank itself");
            }
            if(intersectingRanks.size() == 1)
            {
                return intersectingRanks;
            }
            if(query.maxPointsToGet != 1 or distributedOctAgent == nullptr)
            {
                return intersectingRanks;
            }
            // if the query requests to ask all the intersecting ranks, return all the intersecting ranks
            if(query.whoToAsk == ASK_ALL)
            {
                return intersectingRanks;
            }
            // otherwise, the queries requests to ask only the close ranks
            // we calculate the closest distances from the point, to all the other ranks.
            std::vector<std::pair<double, double>> distances;
            // maybe the distances were already computed (check in a cache)
            auto it = this->resultCache.find(query.pointIndex);
            if(it != this->resultCache.end())
            {
                distances = (*it).second;
            }
            else
            {
                // not in cache, calculate it and insert to the cache
                Vector3D point(query.extraPoint.x, query.extraPoint.y, query.extraPoint.z);
                distances = distributedOctAgent->getOctTree()->getClosestFurthestPointsByRanks(point);
                this->resultCache.insert({query.pointIndex, distances});
            }
            // get the closest rank
            double minDist = std::numeric_limits<double>::max();
            size_t minDistRank = std::numeric_limits<size_t>::max();
            for(const int &_rank : intersectingRanks)
            {
                if(static_cast<int>(_rank) == this->rank)
                {
                    continue; // don't count myself
                }
                if(distances[_rank].first < minDist)
                {
                    minDist = distances[_rank].first;
                    minDistRank = _rank;
                }
            }
            if(minDistRank > static_cast<size_t>(this->size))
            {
                std::cout << "min dist rank is " << minDistRank << std::endl;
                // in fact, should not reach here, if intersectingRanks.size() > 1
                throw UniversalError("In range talk agent, should not reach here (no rank found)");
            }
            // consider the closest rank, and its furthest distance from the point, denoted as `closestDistThreshold`
            double closestDistThreshold = distances[minDistRank].second;

            // return all the ranks which their closest point to us is in distance of at most `closestDistThreshold`
            EnvironmentAgent::RanksSet result;
            for(const int &_rank : intersectingRanks)
            {
                if(distances[_rank].first <= (closestDistThreshold * (1 + EPSILON)))
                {
                    result.insert(_rank);
                }
            }
            return result;
        }

    private:
        const EnvironmentAgent *envAgent;
        mutable _map<size_t, std::vector<std::pair<double, double>>> resultCache;
        int rank, size;
    };

public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    RangeAgent(const EnvironmentAgent *envAgent, RangeFinder *rangeFinder, const std::vector<int> &_sentProc = std::vector<int>(), const std::vector<std::vector<size_t>> &_sentPoints = std::vector<std::vector<size_t>>(),
                const std::vector<int> &_recvProc = std::vector<int>(), const std::vector<std::vector<size_t>> &_recvPoints = std::vector<std::vector<size_t>>(), const MPI_Comm &comm = MPI_COMM_WORLD)
    {
        RangeAnswerAgent *answersAgent = new RangeAnswerAgent(rangeFinder);
        this->ansAgent = answersAgent;
        this->talkAgent = new RangeTalkAgent(envAgent, comm);
        this->queryAgent = new QueryAgent<RangeQueryData, _3DPoint>(this->talkAgent, this->ansAgent, false /* dont send messages to self */, comm);

        std::vector<std::vector<std::size_t>> &sentPoints = this->getSentPoints();
        std::vector<RangeFinder::_set<size_t>> &sentPointsSet = answersAgent->sentDataSet;
        std::vector<std::vector<std::size_t>> &recvPoints = this->getRecvPoints();
        std::vector<int> &sentProc = this->getSentProc();
        std::vector<int> &recvProc = this->getRecvProc();
        
        sentProc.insert(sentProc.end(), _sentProc.begin(), _sentProc.end());
        recvProc.insert(recvProc.end(), _recvProc.begin(), _recvProc.end());
        sentPoints.insert(sentPoints.end(), _sentPoints.begin(), _sentPoints.end());
        recvPoints.insert(recvPoints.end(), _recvPoints.begin(), _recvPoints.end());

        for(size_t i = 0; i < sentPoints.size(); i++)
        {
            sentPointsSet.emplace_back(RangeFinder::_set<size_t>(sentPoints[i].begin(), sentPoints[i].end()));
        }
    }

    ~RangeAgent()
    {
        delete this->queryAgent;
        delete this->talkAgent;
        delete this->ansAgent;
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
    RangeAnswerAgent *ansAgent;
    RangeTalkAgent *talkAgent;
    QueryAgent<RangeQueryData, _3DPoint> *queryAgent;
};

#endif // RICH_MPI

#endif // _RICH_RANGE_AGENT_HPP_