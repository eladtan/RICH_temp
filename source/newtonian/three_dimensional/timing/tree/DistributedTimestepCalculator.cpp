#include "DistributedTimestepCalculator.hpp"

#ifdef RICH_MPI

namespace
{   
    template<typename T>
    bool calculateSendNodesHelper(const typename TimingTree<T>::Node *node, const std::vector<TimeRequestData<T>> &requestData, std::vector<typename TimingTree<T>::NodeData> &sendNodes)
    {
        if(node == nullptr)
        {
            return false;
        }

        bool shouldOpen = false;
        for(const TimeRequestData<T> &request : requestData)
        {
            if(node->boundingBox.intersects(request.boundingBox) or ShouldOpenNode<T>(node->value, request.min_time_in_subtree, request.c_max_plus_v_max, request.boundingBox))
            {
                shouldOpen = true;
                break;
            }
        }

        if(node->isLeaf or !shouldOpen)
        {
            // do not open, just send me
            sendNodes.push_back(node->value);
            return true;
        }
        // call recursively to children. If one (or more) has a sent node in its subtree, send all the children instead of me,
        // otherwise, send merely me
        std::vector<bool> childrenSent(CHILDREN, false);
        bool oneOfChildrenIsSent = false;

        for(int i = 0; i < CHILDREN; i++)
        {
            childrenSent[i] = calculateSendNodesHelper(node->children[i], requestData, sendNodes);
            oneOfChildrenIsSent = (oneOfChildrenIsSent or childrenSent[i]);
        }

        if(!oneOfChildrenIsSent)
        {
            return false;
        }
        else
        {
            // send my unsent children
            for(int i = 0; i < CHILDREN; i++)
            {
                if(node->children[i] != nullptr)
                {
                    if(!childrenSent[i])
                    {
                        sendNodes.push_back(node->children[i]->value);
                    }
                }
            }
            return true;
        }
    }

    template<typename T>
    inline std::vector<typename TimingTree<T>::NodeData> getNecessaryNodes(const TimingTree<T> *tree, const std::vector<TimeRequestData<T>> &requestData)
    {
        std::vector<typename TimingTree<T>::NodeData> sendNodes;
        calculateSendNodesHelper(tree->getOctTree()->getRoot(), requestData, sendNodes);
        return sendNodes;
    }

    template<typename T>
    inline std::vector<std::vector<typename TimingTree<T>::NodeData>> getNecessaryNodesByRanks(const TimingTree<T> *tree, const std::vector<std::vector<TimeRequestData<T>>> &requestData)
    {
        std::vector<std::vector<typename TimingTree<T>::NodeData>> resultByRanks;
        for(const std::vector<TimeRequestData<T>> &requestDataOfRank : requestData)
        {
            resultByRanks.emplace_back(getNecessaryNodes(tree, requestDataOfRank));
        }
        return resultByRanks;
    }
}

DistributedTimestepCalculator::DistributedTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities, const MPI_Comm &comm):
        comm(comm), tess(tess), cells(cells), faceVelocities(faceVelocities), createdTimingTree(true)
{
    MPI_Comm_size(this->comm, &this->size);
    MPI_Comm_rank(this->comm, &this->rank);
    TimingTree<Vector3D> *timingTree = new TimingTree<Vector3D>(ll, ur);
    timingTree->build(tess, cells, faceVelocities);
    this->timingTree = timingTree;
    this->distributedTree = new DistributedOctTree<NodeData, 1>(this->timingTree->getOctTree(), false /* we use nodes directions (need `detailedNodeInfo` for that) */, this->comm);
}

DistributedTimestepCalculator::DistributedTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities, TimingTree<Vector3D> *timingTree, const MPI_Comm &comm):
    comm(comm), tess(tess), cells(cells), faceVelocities(faceVelocities),  createdTimingTree(false)
{
    MPI_Comm_size(this->comm, &this->size);
    MPI_Comm_rank(this->comm, &this->rank);
    this->timingTree = timingTree;
    this->distributedTree = new DistributedOctTree<NodeData, 1>(this->timingTree->getOctTree(), true /* we use nodes directions (need `detailedNodeInfo` for that) */, this->comm);
}
   
DistributedTimestepCalculator::~DistributedTimestepCalculator()
{
    if(this->createdTimingTree)
    {
        delete this->timingTree;
    }
    delete this->distributedTree;
}

/**
    \brief Returns a list of new points to add to my local timing tree, to make it 'global'
*/
std::vector<std::vector<typename DistributedTimestepCalculator::NodeData>> DistributedTimestepCalculator::exchangeImportedValues(void) const
{
    std::vector<TimeRequestData<Vector3D>> myRequestsToAll;
    std::vector<std::vector<direction_t>> myDirections = this->distributedTree->getMyDirections();
    for(const std::vector<direction_t> &direction : myDirections)
    {
        const typename TimingTree<Vector3D>::Node *node = this->timingTree->getOctTree()->getNodeByDirections(direction.data());
        std::cout << "node->value.min_time_in_subtree is " << node->value.min_time_in_subtree << std::endl;
        TimeRequestData<Vector3D> request(node->boundingBox, node->value.min_time_in_subtree, node->value.cell_width);
        myRequestsToAll.push_back(request);
    }
    std::vector<std::vector<TimeRequestData<Vector3D>>> requestedData = MPI_All_cast_by_ranks(myRequestsToAll, this->comm);
    // now analyse `requestedData`. For each bounding box and time, determine what bounding boxes should be sent
    std::vector<std::vector<NodeData>> responseData = getNecessaryNodesByRanks(this->timingTree, requestedData); 
    // exchange result
    std::vector<std::vector<NodeData>> responseAllData = MPI_Exchange_all_to_all(responseData, this->comm);
    return responseAllData;
}

std::vector<dt_t> DistributedTimestepCalculator::calculateTimesForIndices(const std::vector<size_t> &cellsIndices) const
{
    // check all the cells are mine
    for(const size_t &cellIndex : cellsIndices)
    {
        Vector3D point = this->tess.GetMeshPoint(cellIndex);
        if(not this->timingTree->getOctTree()->find(point))
        {
            UniversalError eo("DistributedTimestepCalculator: A point is not owned by rank");
            eo.addEntry("Index", cellIndex);
            eo.addEntry("Point", point);
            eo.addEntry("Rank", rank);
            throw eo;
        }
    }
    
    // add the external values, to make the tree 'global'

    this->timingTree->calculateSelfTimes(cellsIndices);

    std::vector<std::vector<NodeData>> data = this->exchangeImportedValues();
    for(const std::vector<NodeData> &responseOfRank : data)
    {
        this->timingTree->addExternalValues(responseOfRank);
    }

    // calculate the results
    std::vector<dt_t> times;
    for(const size_t &cellIndex : cellsIndices)
    {
        dt_t cellTime = this->timingTree->time(cellIndex);
        this->cells[cellIndex].dt = cellTime;
        times.push_back(cellTime);
    }
    return times;
}

#endif // RICH_MPI