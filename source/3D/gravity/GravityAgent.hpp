#ifndef _GRAVITY_AGENT_HPP
#define _GRAVITY_AGENT_HPP

#ifdef RICH_MPI

#include "utils/queryAgent/QueryAgent.hpp"
#include "DistributedGravityTree.hpp"
#include "newtonian/three_dimensional/ConservativeForce3D.hpp"

#include "GravityTree.hpp"

struct GravityQueryData
{
    _3DPoint point;
    size_t pointIdx;
    gravity_result_t mass;
    GravityTreeLocation location;
};

class GravityAgent : public Acceleration3D
{
private:
    class GravityAnswerAgent : public AnswerAgent<GravityQueryData, _3DPoint>
    {
    public:
        GravityAnswerAgent(const GravityTree<_3DPoint> *gravityTree): gravityTree(gravityTree){};

        inline std::vector<_3DPoint> answer(const GravityQueryData &query, int _rank) override
        {
            return std::vector<_3DPoint>({this->gravityTree->gravity(query.mass, query.point, query.location.directions)});
        }

    private:
        const GravityTree<_3DPoint> *gravityTree;
    };

    class GravityTalkAgent : public TalkAgent<GravityQueryData>
    {
    public:
        GravityTalkAgent(){};

        inline _set<int> getTalkList(const GravityQueryData &query) const override
        {
            return _set<int>({query.location.rank});
        }
    };

public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    GravityAgent(const GravityTree<_3DPoint> *gravityTree, const MPI_Comm &comm = MPI_COMM_WORLD)
    {
        this->distributedGravityTree = new DistributedGravityTree(gravityTree, comm);
        this->ansAgent = new GravityAnswerAgent(gravityTree);
        this->talkAgent = new GravityTalkAgent();
        this->queryAgent = new QueryAgent<GravityQueryData, _3DPoint>(this->talkAgent, this->ansAgent, true /*consider self gravity*/, comm);
    }

    ~GravityAgent() override
    {
        delete this->distributedGravityTree;
        delete this->queryAgent;
        delete this->talkAgent;
        delete this->ansAgent;
    }

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const;

    std::vector<Vector3D> getForces(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses) const;

private:
    DistributedGravityTree<_3DPoint> *distributedGravityTree;
    AnswerAgent<GravityQueryData, _3DPoint> *ansAgent;
    QueryAgent<GravityQueryData, _3DPoint> *queryAgent;
    TalkAgent<GravityQueryData> *talkAgent;
};

std::vector<Vector3D> GravityAgent::getForces(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses) const
{
    std::queue<GravityQueryData> queries;
    std::vector<Vector3D> res(points.size(), Vector3D(0, 0, 0));

    for(size_t i = 0; i < points.size(); i++)
    {
        _3DPoint point(points[i].x, points[i].y, points[i].z);
        // get the list of locations (pairs of <rank, directions in its tree>) we should talk to
        auto forceCalcData = this->distributedGravityTree->getLocationList(point);
        res[i] = Vector3D(forceCalcData.first.x, forceCalcData.first.y, forceCalcData.first.z); // initial force (from unopened boxes)
        for(const GravityTreeLocation &location : forceCalcData.second)
        {
            // add a query to this point
            GravityQueryData data;
            data.point = point;
            data.mass = masses[i];
            data.pointIdx = i;
            data.location = location;
            queries.emplace(data);
        }
    }
    
    // submit the batch, and get the answer
    QueryBatchInfo<GravityQueryData, _3DPoint> batchInfo = this->queryAgent->runBatch(queries);
    std::vector<QueryInfo<GravityQueryData, _3DPoint>> &answers = batchInfo.queriesAnswers;

    for(const QueryInfo<GravityQueryData, _3DPoint> &query : answers)
    {
        size_t pointIdx = query.data.pointIdx;
        const _3DPoint &force =  query.finalResults[0];
        res[pointIdx] += Vector3D(force.x, force.y, force.z); 
    }
    return res;
}

void GravityAgent::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const
{
    std::queue<GravityQueryData> queries;
    
    std::vector<Vector3D> points = tess.getMeshPoints();
    points.resize(tess.GetPointNo());

    std::vector<gravity_result_t> masses;
    masses.reserve(points.size());

    for(size_t i = 0; i < points.size(); i++)
    {
        masses.push_back((cells[i].density) * (tess.GetVolume(i)));
    }
    acc = std::move(this->getForces(points, masses));
    
    /*
    acc.resize(points.size());
    std::fill(acc.begin(), acc.end(), Vector3D(0, 0, 0));

    for(size_t i = 0; i < points.size(); i++)
    {
        _3DPoint point(points[i].x, points[i].y, points[i].z);
        // get the list of locations (pairs of <rank, directions in its tree>) we should talk to
        auto forceCalcData = this->distributedGravityTree->getLocationList(point);
        acc[i] = Vector3D(forceCalcData.first.x, forceCalcData.first.y, forceCalcData.first.z); // initial force (from unopened boxes)
        for(const GravityTreeLocation &location : forceCalcData.second)
        {
            // add a query to this point
            GravityQueryData data;
            data.point = point;
            data.mass = ;
            data.pointIdx = i;
            data.location = location;
            queries.emplace(data);
        }
    }

    // submit the batch, and get the answer
    QueryBatchInfo<GravityQueryData, _3DPoint> batchInfo = this->queryAgent->runBatch(queries);
    std::vector<QueryInfo<GravityQueryData, _3DPoint>> &answers = batchInfo.queriesAnswers;

    for(const QueryInfo<GravityQueryData, _3DPoint> &query : answers)
    {
        size_t pointIdx = query.data.pointIdx;
        const _3DPoint &force =  query.finalResults[0];
        acc[pointIdx] += Vector3D(force.x, force.y, force.z); 
    }
    */
}

#endif // RICH_MPI

#endif // _GRAVITY_AGENT_HPP