#ifndef BIG_RANGE_AGENT_HPP
#define BIG_RANGE_AGENT_HPP

#ifdef RICH_MPI

#include "utils/queryAgent/QueryAgent.hpp"
#include "3D/range/finders/RangeFinder.hpp"
#include "3D/range/finders/utils/IndexedVector.hpp"
#include "3D/environment/EnvironmentAgent.h"
#include "3D/environment/HilbertTreeEnvAgent.hpp"
#include "3D/environment/DistributedOctEnvAgent.hpp" 
#include "SentPointsContainer.hpp"
#include "RangeQueryData.h"

struct BigRangeQueryData : public RangeQueryData
{
    _3DPoint originalPoint;
    bool askOnlyClose; // in a case of a big query, we can ask all the ranks, or only the close ranks 

    friend inline std::ostream &operator<<(std::ostream &stream, const BigRangeQueryData &query)
    {
        return stream << "[BIG, point is " << query.originalPoint << ", sphere is (center = " << query.center << ", r = " << query.radius << ")]";
    }
};

/**
 * The range agent is responsible for running batches of range queries. A batch is a collection of queries, and a range query is an instance of the `RangeQueryData` class, containing a point and a requested radius.
 * The range agent switches between roles - sending queries, receiving answers, and answering for incoming queries. It also supports duplications removal, and returns the results rearranged by processes (what are the points that were received from each one, and what points I sent to each one).
 * In order to answer for incoming requests, a range finder is required. A range finder is an object which holds a list of points, and can answer for range queries.
*/
class BigRangeAgent
{
public:
    using SmartEnvironmentAgent = DistributedOctEnvironmentAgent;

private:
    class RangeAnswerAgent : public AnswerAgent<BigRangeQueryData, _3DPoint>
    {
        friend class RangeAgent;

    public:
        RangeAnswerAgent(const RangeFinder *rangeFinder, SentPointsContainer &pointsContainer, const MPI_Comm &comm = MPI_COMM_WORLD): rangeFinder(rangeFinder), pointsContainer(pointsContainer){}

        std::vector<_3DPoint> answer(const BigRangeQueryData &query, int _rank) override
        {
            // std::cout << "answering to rank " << _rank << " about query " << query << std::endl;            
            const SentPointsContainer::PointsSet &ignore = this->pointsContainer.getSentDataSetRank(_rank);

            // a big query, bring only the closest point
            std::vector<size_t> indicesResult = this->rangeFinder->closestPointInSphere(Vector3D(query.center.x, query.center.y, query.center.z), query.radius, Vector3D(query.originalPoint.x, query.originalPoint.y, query.originalPoint.z), ignore);
            indicesResult = this->pointsContainer.addPointsAsSent<std::vector>(_rank, indicesResult);

            std::vector<_3DPoint> result;
            result.reserve(indicesResult.size());
            for(const size_t &pointIdx : indicesResult)
            {
                result.push_back(_3DPoint(this->rangeFinder->getPoint(pointIdx)));
            }
            return result;
        }

    private:
        const RangeFinder *rangeFinder;
        SentPointsContainer &pointsContainer;
    };

    class RangeTalkAgent : public TalkAgent<BigRangeQueryData>
    {
    public:
        template<typename K, typename V>
        using _map = boost::container::flat_map<K, V>;

        RangeTalkAgent(const EnvironmentAgent *envAgent, const MPI_Comm &comm = MPI_COMM_WORLD): envAgent(envAgent)
        {
            MPI_Comm_rank(comm, &this->rank);
            MPI_Comm_size(comm, &this->size);
        };

        inline EnvironmentAgent::RanksSet getTalkList(const BigRangeQueryData &query) const override
        {
            // std::cout << "rank " << this->rank << " calculates the talk list of query " << query << std::endl;
            EnvironmentAgent::RanksSet intersectingRanks = this->envAgent->getIntersectingRanks(Vector3D(query.center.x, query.center.y, query.center.z), query.radius);
            if(intersectingRanks.empty())
            {
                throw UniversalError("In range talk agent, should not reach here: the intersecting ranks list should at least contain the rank itself");
            }
            if(intersectingRanks.size() == 1)
            {
                return intersectingRanks;
            }

            // check if has 'smartAgent' (an agent that can caluclate distances of ranks as well)
            const SmartEnvironmentAgent *smartAgent = dynamic_cast<const SmartEnvironmentAgent*>(this->envAgent);
            if(smartAgent == nullptr)
            {
                return intersectingRanks;
            }

            // if the query requests to ask all the intersecting ranks, return all the intersecting ranks
            if(not query.askOnlyClose)
            {
                return intersectingRanks; // ask all
            }

            // otherwise, the queries requests to ask only the close ranks
            // we calculate the closest distances from the point, to all the other ranks.
            std::vector<std::pair<double, double>> distances;
            // maybe the distances were already computed (check in a cache)
            auto it = this->resultCache.find(query.pointIdx);
            if(it != this->resultCache.end())
            {
                distances = (*it).second;
            }
            else
            {
                // not in cache, calculate it and insert to the cache
                distances = smartAgent->getClosestFurthestPointsByRanks(query.originalPoint);
                this->resultCache.insert(std::pair<size_t, decltype(distances)>({query.pointIdx, distances}));
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

    BigRangeAgent(const EnvironmentAgent *envAgent, RangeFinder *rangeFinder, SentPointsContainer &pointsContainer, const MPI_Comm &comm = MPI_COMM_WORLD): pointsContainer(pointsContainer)
    {
        RangeAnswerAgent *answersAgent = new RangeAnswerAgent(rangeFinder, pointsContainer);
        this->ansAgent = answersAgent;
        this->talkAgent = new RangeTalkAgent(envAgent, comm);
        this->queryAgent = new QueryAgent<BigRangeQueryData, _3DPoint>(this->talkAgent, this->ansAgent, false /* dont send messages to self */, comm);
    }

    ~BigRangeAgent()
    {
        delete this->queryAgent;
        delete this->talkAgent;
        delete this->ansAgent;
    }

    inline QueryBatchInfo<BigRangeQueryData, _3DPoint> runBatch(std::queue<BigRangeQueryData> &queries)
    {
        return this->queryAgent->runBatch(queries);
    };

    inline std::vector<std::vector<std::size_t>> &getSentPoints(){return this->pointsContainer.getSentData();};
    inline std::vector<std::vector<std::size_t>> &getRecvPoints(){return this->queryAgent->getRecvData();};
    inline std::vector<int> &getSentProc(){return this->pointsContainer.getSentProc();};
    inline std::vector<int> &getRecvProc(){return this->queryAgent->getRecvProc();};

private:
    RangeAnswerAgent *ansAgent;
    RangeTalkAgent *talkAgent;
    QueryAgent<BigRangeQueryData, _3DPoint> *queryAgent;
    SentPointsContainer &pointsContainer;
};

#endif // RICH_MPI

#endif // BIG_RANGE_AGENT_HPP