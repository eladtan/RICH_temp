#ifdef RICH_MPI

#include "RangeAgent.h"

RangeAgent::RangeAgent(const EnvironmentAgent *envAgent, RangeFinder *rangeFinder, const MPI_Comm &comm):
        comm(comm), envAgent(envAgent), rangeFinder(rangeFinder)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->ranksBufferIdx.resize(this->size, UNDEFINED_BUFFER_IDX);
}

void RangeAgent::receiveQueries(QueryBatchInfo &batch)
{
    if(this->receivedUntilNow >= this->shouldReceiveInTotal)
    {
        return;
    }
    MPI_Status status;
    int receivedAnswer = 0;
    int received = 0;

    std::vector<QueryInfo> &queries = batch.queriesAnswers;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESPONSE, this->comm, &receivedAnswer, &status);

    std::vector<char> buffer;

    while(receivedAnswer and (received < MAX_RECEIVE_IN_CYCLE))
    {
        // received a message
        ++this->receivedUntilNow;
        ++received;

        // prepare the reading buffer for receiving
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        if(buffer.size() < static_cast<size_t>(count))
        {
            buffer.resize(count);
        }

        // receive
        MPI_Recv(&buffer[0], count, MPI_BYTE, status.MPI_SOURCE, TAG_RESPONSE, this->comm, MPI_STATUS_IGNORE);

        // decode the message - id first, then length, then the data itself
        long int id;
        long int length;
        int pos = 0;

        id = *reinterpret_cast<long int*>(buffer.data()); // decode id
        length = *reinterpret_cast<long int*>(buffer.data() + sizeof(long int)); // decode length

        if(length > 0)
        {
            // insert the results to the points received by rank `status.MPI_SOURCE` and to the queries result
            queries[id].finalResults.resize(queries[id].finalResults.size() + length);

            // base pointers for points data
            _3DPoint* base = reinterpret_cast<_3DPoint*>(buffer.data() + 2 * sizeof(long int));
            for(size_t i = 0; i < static_cast<size_t>(length); i++)
            {
                _3DPoint &point = base[i]; // same as `*(queries[id].finalResults.end() - length + i)`;
                batch.newPointsRanks[status.MPI_SOURCE].emplace_back(Vector3D(point.x, point.y, point.z));
            }
        }
        else
        {
            assert(length >= 0);
        }
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESPONSE, this->comm, &receivedAnswer, &status);
    }
}

/**
 * gets a query, and who requests it, and returns the answer that should be sent (the result should be clean from duplications, that is, points we sent before).
*/
std::vector<Vector3D> RangeAgent::getRangeResult(const SubQueryData &query, int rank)
{
    std::vector<Vector3D> result;
    // get the real results, and filter it (do not send points you sent before)
    std::vector<size_t> nonFilteredResult;
    if(query.data.maxPointsToGet == 1)
    {
        size_t rankIndex = std::find(this->sentProcessorsRanks.begin(), this->sentProcessorsRanks.end(), rank) - this->sentProcessorsRanks.begin();
        const _set<size_t> &ignore = (rankIndex == this->sentProcessorsRanks.size())? _set<size_t>() : this->sentPointsSet[rankIndex];
        nonFilteredResult = this->rangeFinder->closestPointInSphere(Vector3D(query.data.center.x, query.data.center.y, query.data.center.z), query.data.radius, Vector3D(query.data.extraPoint.x, query.data.extraPoint.y, query.data.extraPoint.z), ignore);
    }
    else
    {
        nonFilteredResult = this->rangeFinder->range(Vector3D(query.data.center.x, query.data.center.y, query.data.center.z), query.data.radius, query.data.maxPointsToGet);
    }
    if(!nonFilteredResult.empty())
    {
        // get what is the right index of `node` inside the sentProcessors vector. If it isn't there, create it
        size_t rankIndex = std::find(this->sentProcessorsRanks.begin(), this->sentProcessorsRanks.end(), rank) - this->sentProcessorsRanks.begin();
        if(rankIndex == this->sentProcessorsRanks.size())
        {
            // rank is not inside the sentProcessors rank, add it
            this->sentProcessorsRanks.push_back(rank);
            this->sentPointsSet.push_back(_set<size_t>());
            this->sentPoints.push_back(std::vector<size_t>());
        }
        for(const size_t &pointIdx : nonFilteredResult)
        {
            // check if the point wasn't already sent to node, only after that, add it to the result
            if(this->sentPointsSet[rankIndex].find(pointIdx) == this->sentPointsSet[rankIndex].end())
            {
                // point haven't been sent, send it
                result.push_back(this->rangeFinder->getPoint(pointIdx));
                this->sentPointsSet[rankIndex].insert(pointIdx);
                this->sentPoints[rankIndex].push_back(pointIdx);
            }
        }
    }
    return result;
}

void RangeAgent::answerQueries()
{
    static int totalArrived = 0;

    MPI_Status status;
    int arrivedNew = 0;
    int answered = 0;

    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, this->comm, &arrivedNew, &status);
    
    std::vector<char> arriveBuffer;

    // while arrived new messages, and we should answer until the end, or answer until a bound we haven't reached to
    while((arrivedNew != 0) and (answered < MAX_ANSWER_IN_CYCLE))
    {
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        if(arriveBuffer.size() < static_cast<size_t>(count))
        {
            arriveBuffer.resize(count);
        }
        MPI_Recv(&arriveBuffer[0], count, MPI_BYTE, status.MPI_SOURCE, TAG_REQUEST, this->comm, MPI_STATUS_IGNORE);
        int subQueries = count / sizeof(SubQueryData);
        totalArrived += subQueries;
        for(int i = 0; i < subQueries; i++)
        {
            const SubQueryData &query = *reinterpret_cast<SubQueryData*>(&arriveBuffer[i * sizeof(SubQueryData)]);
            answered++;
            // calculate the result
            std::vector<Vector3D> result = this->getRangeResult(query, status.MPI_SOURCE);
            long int resultSize = static_cast<long int>(result.size());
            this->buffers.push_back(std::vector<char>());
            std::vector<char> &to_send = this->buffers[this->buffers.size() - 1];
            size_t msg_size = 2 * sizeof(long int) + resultSize * sizeof(_3DPoint);
            to_send.resize(msg_size);

            long int id = query.parent_id;

            int pos = 0;
            /*
            MPI_Pack(&id, 1, MPI_LONG, &to_send[0], msg_size, &pos, this->comm);
            MPI_Pack(&resultSize, 1, MPI_LONG, &to_send[0], msg_size, &pos, this->comm);
            */
           *reinterpret_cast<long int*>(to_send.data()) = id;
           *reinterpret_cast<long int*>(to_send.data() + sizeof(long int)) = resultSize;

            if(resultSize > 0)
            {
                _3DPoint *points = reinterpret_cast<_3DPoint*>(to_send.data() + sizeof(id) + sizeof(resultSize));
                for(size_t i = 0; i < static_cast<size_t>(resultSize); i++)
                {
                    points[i] = {result[i].x, result[i].y, result[i].z};
                }
            }

            /*
            this->requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(&to_send[0], msg_size, MPI_BYTE, status.MPI_SOURCE, TAG_RESPONSE, this->comm, &this->requests[requests.size() - 1]);
            */
            MPI_Send(&to_send[0], msg_size, MPI_BYTE, status.MPI_SOURCE, TAG_RESPONSE, this->comm);
        }
        MPI_Iprobe(MPI_ANY_SOURCE,  TAG_REQUEST, this->comm, &arrivedNew, &status);
    }
}

void RangeAgent::sendQuery(const QueryInfo &query)
{
    _set<int> possibleNodes = this->envAgent->getIntersectingRanks(Vector3D(query.data.center.x, query.data.center.y, query.data.center.z), query.data.radius);
    for(const int &node : possibleNodes)
    {
        if(node == this->rank)
        {
            continue; // unnecessary to send
        }
        int bufferIdx = this->ranksBufferIdx[node];
        // check if a flush is needed

        if((bufferIdx != UNDEFINED_BUFFER_IDX) and ((this->buffers[bufferIdx].size() / sizeof(SubQueryData)) >= FLUSH_QUERIES_NUM))
        {
            // send buffer
            this->flushBuffer(node);
        }
        bufferIdx = this->ranksBufferIdx[node];
        if(bufferIdx == UNDEFINED_BUFFER_IDX)
        {
            this->buffers.push_back(std::vector<char>());
            this->buffers[this->buffers.size() - 1].reserve(sizeof(SubQueryData) * FLUSH_QUERIES_NUM);
            this->ranksBufferIdx[node] = this->buffers.size() - 1;
        }
        bufferIdx = this->ranksBufferIdx[node];
        this->buffers[bufferIdx].resize(this->buffers[bufferIdx].size() + sizeof(SubQueryData));
        SubQueryData &subQuery = *reinterpret_cast<SubQueryData*>(&(*(this->buffers[bufferIdx].end() - sizeof(SubQueryData))));
        subQuery.data = query.data;
        subQuery.parent_id = query.id;
        ++this->shouldReceiveInTotal;
    }
}

void RangeAgent::sendFinish()
{
    int dummy;
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        this->requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&dummy, 1, MPI_BYTE, _rank, TAG_FINISHED, this->comm, &this->requests[this->requests.size() - 1]);
    }
}

int RangeAgent::checkForFinishMessages() const
{
    int arrived = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_FINISHED, this->comm, &arrived, &status);
    if(arrived)
    {
        int dummy = 0;
        MPI_Recv(&dummy, 1, MPI_BYTE, MPI_ANY_SOURCE, TAG_FINISHED, this->comm, MPI_STATUS_IGNORE);
        return 1;
    }
    return 0;
}

void RangeAgent::flushBuffer(int node)
{
    int bufferIdx = this->ranksBufferIdx[node];
    if(bufferIdx == UNDEFINED_BUFFER_IDX)
    {
        return;
    }
    if(this->buffers[bufferIdx].size() > 0)
    {
        this->requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&this->buffers[bufferIdx][0], this->buffers[bufferIdx].size(), MPI_BYTE, node, TAG_REQUEST, this->comm, &this->requests[this->requests.size() - 1]);
    }
    this->ranksBufferIdx[node] = UNDEFINED_BUFFER_IDX;
}

QueryBatchInfo RangeAgent::runBatch(std::queue<RangeQueryData> &queries)
{
    this->receivedUntilNow = 0; // reset the receive counter
    this->shouldReceiveInTotal = 0; // reset the should-be-received counter
    for(std::vector<size_t> &_rankPoints : this->recvPoints)
    {
        _rankPoints.clear();
    }

    this->buffers.clear();
    size_t originalQueriesNum = queries.size();
    this->buffers.reserve(10 * originalQueriesNum); // heuristic
    this->requests.clear();
    QueryBatchInfo queriesBatch;
    std::vector<QueryInfo> &queriesInfo = queriesBatch.queriesAnswers;
    queriesBatch.newPointsRanks.resize(this->size);
    int finishedReceived = 0;
    bool sentFinished = false;
    size_t i = 0;
    bool notEmpty = false;

    while((notEmpty = (i < originalQueriesNum)) or (finishedReceived < this->size))
    {
        if(notEmpty)
        {
            RangeQueryData queryData = queries.front();
            queries.pop();
            queriesInfo.push_back({queryData, i, std::vector<_3DPoint>()});
            QueryInfo &query = queriesInfo[queriesInfo.size() - 1];
            this->sendQuery(query);
            if(i == (originalQueriesNum - 1))
            {
                // send the rest of the waiting (buffered) requests
                this->flushAll();
            }
        }
        else
        {
            if(i % FINISH_AUTOFLUSH_NUM == 0 and this->shouldReceiveInTotal == this->receivedUntilNow)
            {
                if(!sentFinished)
                {
                    sentFinished = true;
                    this->sendFinish();
                }
                finishedReceived += this->checkForFinishMessages();
            }
        }
        if(this->shouldReceiveInTotal > this->receivedUntilNow)
        {
            // std::cout << "recieved " << this->receivedUntilNow << " out of " << this->shouldReceiveInTotal << std::endl;
            this->receiveQueries(queriesBatch);
        }

        if(i % QUERY_AUTOFLUSH_NUM == 0)
        {
            this->answerQueries();
        }
        ++i;
    }
    if(!this->requests.empty())
    {
        MPI_Waitall(this->requests.size(), &this->requests[0], MPI_STATUSES_IGNORE); // make sure any query was indeed received
    }

    // add to the list the processors that sent us a message for the first time
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        if(_rank == this->rank)
        {
            continue; // shouldn't be relevant
        }
        if(queriesBatch.newPointsRanks[_rank].empty())
        {
            continue; // the rank `_rank` did not send us any message
        }
        size_t rankIndex = std::find(this->recvProcessorsRanks.begin(), this->recvProcessorsRanks.end(), _rank) - this->recvProcessorsRanks.begin();
        if(rankIndex == this->recvProcessorsRanks.size())
        {
            // rank is not inside the recvProcessors rank, add it
            this->recvProcessorsRanks.push_back(_rank);
            this->recvPoints.push_back(std::vector<size_t>());
        }
    }

    // make receive, by the order of the recv array
    for(size_t i = 0; i < this->recvProcessorsRanks.size(); i++)
    {
        int _rank = this->recvProcessorsRanks[i];
        std::vector<size_t> &rankRecvPoints = this->recvPoints[i]; // indices vector
        std::vector<Vector3D> &newPointsFromRank = queriesBatch.newPointsRanks[_rank]; // the points themself
        rankRecvPoints.reserve(newPointsFromRank.size());
        queriesBatch.newPoints.reserve(queriesBatch.newPoints.size() + rankRecvPoints.size());
        for(const Vector3D &_point : newPointsFromRank)
        {
            rankRecvPoints.push_back(queriesBatch.newPoints.size()); // the index of the point received by this rank
            queriesBatch.newPoints.emplace_back(_point);
        }        
    }
    return queriesBatch;
}

#endif // RICH_MPI