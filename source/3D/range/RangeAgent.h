#ifndef _RICH_RANGE_AGENT_H_
#define _RICH_RANGE_AGENT_H_

#ifdef RICH_MPI

#include <iostream> // todo remove
#include <algorithm>
#include <cmath>
#include <set>
#include <queue>
#include <vector>
#include <mpi.h>

// set data structure:
#include <boost/container/flat_set.hpp>
#include <unordered_set>

#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "../environment/EnvironmentAgent.h"
#include "finders/RangeFinder.hpp"

#define TAG_REQUEST 200
#define TAG_RESPONSE 201
#define TAG_FINISHED 202

#define UNDEFINED_BUFFER_IDX -1
#define FLUSH_QUERIES_NUM 80

#define QUERY_AUTOFLUSH_NUM 100
#define RECEIVE_AUTOFLUSH_NUM 15
#define FINISH_AUTOFLUSH_NUM 100
#define MAX_RECEIVE_IN_CYCLE 20
#define MAX_ANSWER_IN_CYCLE 20

#define NO_MAX_POINTS 1000000000

typedef struct RangeQueryData
{
    size_t pointIndex;
    _3DPoint center;
    _3DPoint extraPoint;
    coord_t radius;
    size_t maxPointsToGet;
} RangeQueryData;

typedef struct SubQueryData
{
    RangeQueryData data;
    size_t parent_id;
} SubQueryData;

typedef struct QueryInfo
{
    RangeQueryData data;
    size_t id;
    // int subQueriesNum;
    std::vector<_3DPoint> finalResults;
} QueryInfo;

typedef struct QueryBatchInfo
{
    std::vector<QueryInfo> queriesAnswers;
    std::vector<Vector3D> newPoints;
    std::vector<std::vector<Vector3D>> newPointsRanks;
} QueryBatchInfo;


/**
 * The range agent is responsible for running batches of range queries. A batch is a collection of queries, and a range query is an instance of the `RangeQueryData` class, containing a point and a requested radius.
 * The range agent switches between roles - sending queries, receiving answers, and answering for incoming queries. It also supports duplications removal, and returns the results rearranged by processes (what are the points that were received from each one, and what points I sent to each one).
 * In order to answer for incoming requests, a range finder is required. A range finder is an object which holds a list of points, and can answer for range queries.
*/
class RangeAgent
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    RangeAgent(const EnvironmentAgent *envAgent, RangeFinder *rangeFinder, const MPI_Comm &comm = MPI_COMM_WORLD);

    void receiveQueries(QueryBatchInfo &batch);
    void answerQueries();
    void sendQuery(const QueryInfo &query);
    QueryBatchInfo runBatch(std::queue<RangeQueryData> &queries);

    inline size_t getNumPoints() const{return this->rangeFinder->size();};

    std::vector<std::vector<size_t>> &getSentPoints(){return this->sentPoints;};
    std::vector<std::vector<size_t>> &getRecvPoints(){return this->recvPoints;};
    std::vector<int> &getSentProc(){return this->sentProcessorsRanks;};
    std::vector<int> &getRecvProc(){return this->recvProcessorsRanks;};

private:
    MPI_Comm comm;
    int rank, size;
    std::vector<MPI_Request> requests;
    std::vector<std::vector<char>> buffers; // send buffers, so that they will not be allocated on the stack
    size_t receivedUntilNow; // number of answers I received until now
    size_t shouldReceiveInTotal; // number of answers I have to receive (to know when to finish)
    const EnvironmentAgent *envAgent; // an environmental agent
    const RangeFinder *rangeFinder; // a range finder object

    std::vector<int> sentProcessorsRanks; // a vector of ranks, that we sent points to
    std::vector<int> recvProcessorsRanks;  // a vector of ranks, that we received points from
    std::vector<std::vector<size_t>> recvPoints; // a vector of vectors. The vector in index `i` contains the points indices (relatively to my final answer of the batch) that sent to `this->sentProcessorsRanks[i]`.
    std::vector<std::vector<size_t>> sentPoints; // a vector of vectors. The vector in index `i` contains the points indices (relatively to my own points in the finder) that sent to `this->sentProcessorsRanks[i]`.
    std::vector<_set<size_t>> sentPointsSet; // same as `this->sentPoints`, but uses set for quick search.
    std::vector<size_t> ranksBufferIdx; // maps each rank to an index `i`, where this->buffers[i] contains the prepared buffer for sending to the rank (requests are sent in chunks).

    std::vector<Vector3D> getRangeResult(const SubQueryData &query, int rank);
    _set<int> getIntersectingRanks(const Vector3D &center, coord_t radius) const;
    void sendFinish();
    int checkForFinishMessages() const;
    void flushBuffer(int node);
    inline void flushAll(){for(int i = 0; i < this->size; i++) this->flushBuffer(i);};
};

#endif // RICH_MPI

#endif // _RICH_RANGE_AGENT_H_
