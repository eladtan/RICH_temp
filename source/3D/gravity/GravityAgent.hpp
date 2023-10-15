#ifndef _GRAVITY_AGENT_HPP
#define _GRAVITY_AGENT_HPP

#ifdef RICH_MPI

#include "utils/queryAgent/QueryAgent.hpp"
#include "DistributedGravityTree.hpp"

#include "GravityTree.hpp"

#define MAX_IN_BATCH 1000000

struct GravityQueryData
{
    _3DPoint point;
    size_t pointIdx;
    gravity_result_t mass;
    GravityTreeLocation location;
};

class GravityAgent
{
private:
    class GravityAnswerAgent : public AnswerAgent<GravityQueryData, _3DPoint>
    {
    public:
        GravityAnswerAgent(const GravityTree<_3DPoint> *gravityTree): gravityTree(gravityTree){};

        inline std::vector<_3DPoint> answer(const GravityQueryData &query, int _rank) override
        {
            return std::vector<_3DPoint>({this->gravityTree->gravity(query.point, query.location.directions)});
        }

    private:
        const GravityTree<_3DPoint> *gravityTree;
    };

    class GravityTalkAgent : public TalkAgent<GravityQueryData>
    {
    public:
        GravityTalkAgent(){};

        inline TalkAgent::RanksSet getTalkList(const GravityQueryData &query) const override
        {
            return RanksSet({query.location.rank});
        }
    };

public:
    GravityAgent(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses, const Vector3D &ll, const Vector3D &ur, double theta, bool quadrupole = false, const MPI_Comm &comm = MPI_COMM_WORLD):
            comm(comm), gravityTreeCreated(true)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
        GravityTree<_3DPoint> *gravTree = new GravityTree<_3DPoint>(_3DPoint(ll), _3DPoint(ur), theta, quadrupole);
        std::vector<MassedPoint<_3DPoint>> massedPoints;
        massedPoints.reserve(points.size());
        for(size_t pointIdx = 0; pointIdx < points.size(); pointIdx++)
        {
            massedPoints.emplace_back(MassedPoint<_3DPoint>(points[pointIdx], masses[pointIdx]));
        }
        gravTree->build(massedPoints);
        this->gravityTree = gravTree;
        this->initialize();
    }

    GravityAgent(const GravityTree<_3DPoint> *gravityTree, const MPI_Comm &comm = MPI_COMM_WORLD):
             comm(comm), gravityTreeCreated(false)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
        this->gravityTree = gravityTree;
        this->initialize();
    }

    ~GravityAgent()
    {
        if(this->gravityTreeCreated)
        {
            delete this->gravityTree;
        }
        delete this->distributedGravityTree;
        delete this->queryAgent;
        delete this->talkAgent;
        delete this->ansAgent;
    }

    std::vector<Vector3D> getForces(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses) const;

private:
    MPI_Comm comm;
    int rank, size;
    const GravityTree<_3DPoint> *gravityTree;
    bool gravityTreeCreated; // if the gravity tree should be deleted at the end
    DistributedGravityTree<_3DPoint> *distributedGravityTree;
    AnswerAgent<GravityQueryData, _3DPoint> *ansAgent;
    QueryAgent<GravityQueryData, _3DPoint> *queryAgent;
    TalkAgent<GravityQueryData> *talkAgent;

    inline void initialize()
    {
        this->distributedGravityTree = new DistributedGravityTree(this->gravityTree, this->gravityTree->getQuadrupole(), this->comm);
        this->ansAgent = new GravityAnswerAgent(this->gravityTree);
        this->talkAgent = new GravityTalkAgent();
        this->queryAgent = new QueryAgent<GravityQueryData, _3DPoint>(this->talkAgent, this->ansAgent, false /* don't send to self */, this->comm);
    }
};

std::vector<Vector3D> GravityAgent::getForces(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses) const
{
    size_t pointsSize = points.size();
    std::vector<Vector3D> res(points.size(), Vector3D(0, 0, 0));

    MPI_Request moreRoundRequest;
    size_t pointIdx = 0;

    // the number of queries can be extremly large, so divide the batches into smaller sub-batches
    while(true)
    {
        std::queue<GravityQueryData> queries;
        size_t numQueries = 0;

        while((numQueries < MAX_IN_BATCH) and (pointIdx < pointsSize))
        {
            _3DPoint point(points[pointIdx].x, points[pointIdx].y, points[pointIdx].z);
            // get the list of locations (pairs of <rank, directions in its tree>) we should talk to
            auto forceCalcData = this->distributedGravityTree->getLocationList(point);
            res[pointIdx] = Vector3D(forceCalcData.first.x, forceCalcData.first.y, forceCalcData.first.z); // initial force (from unopened boxes)
            for(const GravityTreeLocation &location : forceCalcData.second)
            {
                if(location.rank == this->rank)
                {
                    // self gravity
                    _3DPoint gravityResult = this->gravityTree->gravity(point, location.directions);
                    res[pointIdx] += Vector3D(gravityResult.x, gravityResult.y, gravityResult.z);
                }
                else
                {
                    // add a query to this point
                    GravityQueryData data;
                    data.point = point;
                    data.mass = masses[pointIdx];
                    data.pointIdx = pointIdx;
                    data.location = location;
                    queries.emplace(data);
                    numQueries++;
                }
            }
            pointIdx++;
        }
        
        // submit the batch, and get the answer
        QueryBatchInfo<GravityQueryData, _3DPoint> batchInfo = this->queryAgent->runBatch(queries);
        std::vector<QueryInfo<GravityQueryData, _3DPoint>> &answers = batchInfo.queriesAnswers;
        int I_finished = (pointIdx == pointsSize);
        int finishedNumber;
        MPI_Iallreduce(&I_finished, &finishedNumber, 1, MPI_INT, MPI_SUM, this->comm, &moreRoundRequest);
        
        for(const QueryInfo<GravityQueryData, _3DPoint> &query : answers)
        {
            size_t pointIdx = query.data.pointIdx;
            const _3DPoint &force =  query.finalResults[0];
            res[pointIdx] += Vector3D(force.x, force.y, force.z); 
        }
        
        // check if someone hasn't finished
        MPI_Wait(&moreRoundRequest, MPI_STATUS_IGNORE);
        if(finishedNumber == this->size)
        {
            break;
        }
    }
    return res;
}

#endif // RICH_MPI

#endif // _GRAVITY_AGENT_HPP