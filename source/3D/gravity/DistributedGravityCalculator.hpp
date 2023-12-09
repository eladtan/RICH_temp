#ifndef _NEW_GRAVITY_AGENT_HPP
#define _NEW_GRAVITY_AGENT_HPP

#ifdef RICH_MPI
#include <mpi.h>
#include "GravityTree.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "3D/hilbert/hilbertTypes.h" // for _3DPoint

#define GRAVITY_POINTS_REQUEST_TAG 1104
#define GRAVITY_POINTS_SEND_TAG 1105

namespace
{
    template<typename T>
    bool calculateSendNodesHelper(const typename GravityTree<T>::Node *node, const std::vector<_BoundingBox<T>> &boundingBoxes, std::vector<typename GravityTree<T>::MassedValue> &sendNodes, double thetaSquared)
    {
        if(node == nullptr)
        {
            return false;
        }

        bool shouldOpen = false;
        for(const _BoundingBox<T> &boundingBox : boundingBoxes)
        {
            T closestPoint = boundingBox.closestPoint(node->value.CM);
            if(node->boundingBox.intersects(boundingBox) or ShouldOpenBox(closestPoint, node->boundingBox, node->value.CM, thetaSquared))
            {
                shouldOpen = true;
                break;
            }
        }

        if(node->isLeaf or !shouldOpen)
        {
            // do not open, send me
            sendNodes.push_back(node->value);
            return true;
        }
        // call recursively to children. If one (or more) has a sent node in its subtree, send all the children instead of me,
        // otherwise, send merely me
        std::vector<bool> childrenSent(CHILDREN, false);
        bool oneOfChildrenIsSent = false;

        for(int i = 0; i < CHILDREN; i++)
        {
            childrenSent[i] = calculateSendNodesHelper(node->children[i], boundingBoxes, sendNodes, thetaSquared);
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
    std::vector<typename GravityTree<T>::MassedValue> getNecessaryNodes(const GravityTree<T> *tree, const std::vector<_BoundingBox<T>> &boundingBoxes)
    {
        double thetaSquared = tree->getTheta() * tree->getTheta();
        std::vector<typename GravityTree<T>::MassedValue> sendNodes;
        calculateSendNodesHelper(tree->getOctTree()->getRoot(), boundingBoxes, sendNodes, thetaSquared);
        return sendNodes;
    }
}
class DistributedGravityCalculator
{
public:
    using MassedValue = GravityTree<_3DPoint>::MassedValue;
public:
    DistributedGravityCalculator(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses, const Vector3D &ll, const Vector3D &ur, double theta, bool quadrupole = false, const MPI_Comm &comm = MPI_COMM_WORLD):
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
        this->distributedTree = new DistributedOctTree<MassedValue, 1>(this->gravityTree->getOctTree(), false /* no need to copy values of leaves */, this->comm);
    }

    DistributedGravityCalculator(GravityTree<_3DPoint> *gravityTree, const MPI_Comm &comm = MPI_COMM_WORLD): comm(comm), gravityTreeCreated(false)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
        this->gravityTree = gravityTree;
        this->distributedTree = new DistributedOctTree<MassedValue, 1>(this->gravityTree->getOctTree(), false /* no need to copy values of leaves */, this->comm);
    }
   
    ~DistributedGravityCalculator()
    {
        if(this->gravityTreeCreated)
        {
            delete this->gravityTree;
        }
        delete this->distributedTree;
    }

    std::vector<Vector3D> getAcceleration(const std::vector<Vector3D> &points) const;

private:
    MPI_Comm comm;
    int rank, size;
    GravityTree<_3DPoint> *gravityTree;
    bool gravityTreeCreated; // if the gravity tree should be deleted at the end
    const DistributedOctTree<MassedValue, 1> *distributedTree;

    std::vector<MassedValue> exchangeImportedValues(const std::vector<Vector3D> &points) const;
};

/**
    \brief Returns a list of new points to add to my local gravity tree, to make it 'global'
*/
std::vector<typename DistributedGravityCalculator::MassedValue> DistributedGravityCalculator::exchangeImportedValues(const std::vector<Vector3D> &points) const
{
    std::vector<MPI_Request> requests;
    requests.reserve(2 * this->size);

    // send everyone my bounding boxes
    std::vector<_3DPoint> boundingBoxesToSend;
    for(const _BoundingBox<MassedValue> &boundingBox : this->distributedTree->getMyBoundingBoxes())
    {
        boundingBoxesToSend.push_back(boundingBox.getLL().value);
        boundingBoxesToSend.push_back(boundingBox.getUR().value);
    }

    for(int _rank = 0; _rank < this->size; _rank++)
    {
        if(_rank == this->rank)
        {
            continue;
        }
        else
        {
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(&boundingBoxesToSend[0], boundingBoxesToSend.size() * sizeof(_3DPoint), MPI_BYTE, _rank, GRAVITY_POINTS_REQUEST_TAG, this->comm, &requests[requests.size() - 1]);
        }
    }

    // get necessary points to send to each rank, and send them
    std::vector<std::vector<GravityTree<_3DPoint>::MassedValue>> sendPoints;
    sendPoints.resize(this->size);
    int requestsReceived = 0;
    while(requestsReceived != (this->size - 1))
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, GRAVITY_POINTS_REQUEST_TAG, this->comm, &status);
        int _rank = status.MPI_SOURCE;
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        std::vector<_3DPoint> receivedPoints(count / sizeof(_3DPoint));
        MPI_Recv(&receivedPoints[0], count, MPI_BYTE, _rank, GRAVITY_POINTS_REQUEST_TAG, this->comm, MPI_STATUS_IGNORE);
        std::vector<_BoundingBox<_3DPoint>> boundingBoxes; // the bounding boxes received
        // interpret the message (every two elements are a bounding box)
        for(size_t i = 0; i < receivedPoints.size(); i += 2)
        {
            boundingBoxes.emplace_back(_BoundingBox<_3DPoint>(receivedPoints[i], receivedPoints[i + 1]));
        }
        sendPoints[_rank] = getNecessaryNodes(this->gravityTree, boundingBoxes);
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&sendPoints[_rank][0], sendPoints[_rank].size() * sizeof(MassedValue), MPI_BYTE, _rank, GRAVITY_POINTS_SEND_TAG, this->comm, &requests[requests.size() - 1]);
        requestsReceived++;
    }

    // receive from other ranks the necessary values, add them to `toAdd`
    std::vector<MassedValue> toAdd;
    int answersReceived = 0;
    while(answersReceived != (this->size - 1))
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, GRAVITY_POINTS_SEND_TAG, this->comm, &status);
        int _rank = status.MPI_SOURCE;
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        size_t numPoints = count / sizeof(MassedValue);
        toAdd.resize(toAdd.size() + numPoints);
        MPI_Recv(&toAdd[toAdd.size() - numPoints], count, MPI_BYTE, _rank, GRAVITY_POINTS_SEND_TAG, this->comm, MPI_STATUS_IGNORE);
        answersReceived++;
    }

    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
    return toAdd;
}

std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
    /*
    the points must ALL reside in this rank's domain, since otherwise a bounding box of a point can be
    imported from another rank, making the gravity calculation incorrect. For instance, consider the following case:
    a point P belongs to rank 1, but its gravity is calculated by 0. Rank 1 might send a bounding box of P to rank 0,
    which will be considered as a new point (with a CM and a mass) in rank 0 - Thus the computation of P, which is
    being done locally on rank 0, will consider the gravitation force of P over itself.
    */
    for(const Vector3D &point : points)
    {
        if(not this->gravityTree->getOctTree()->find(point))
        {
            UniversalError eo("DistributedGravityCalculator: A point is not owned by rank");
            eo.addEntry("Point", point);
            eo.addEntry("Rank", rank);
            throw eo;
        }
    }
    
    // add the external values, to make the tree 'global'
    this->gravityTree->addExternalValues(this->exchangeImportedValues(points));

    // calculate the results
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
    {
        _3DPoint result = this->gravityTree->gravity(_3DPoint(point));
        results.emplace_back(Vector3D(result.x, result.y, result.z));
    }
    return results;
}

#endif // RICH_MPI

#endif // _NEW_GRAVITY_AGENT_HPP