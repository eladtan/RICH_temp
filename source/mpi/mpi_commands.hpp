#ifndef MPI_COMMANDS_HPP
#define MPI_COMMANDS_HPP 1

#ifdef RICH_MPI

#include <vector>
#include <chrono>
#include <mpi.h>
#include <functional>
#include "newtonian/two_dimensional/computational_cell_2d.hpp"
#include "newtonian/two_dimensional/extensive.hpp"
#include "tessellation/tessellation.hpp"
#include "misc/serializable.hpp"
#include "misc/utils.hpp"
#include "3D/tesselation/Tessellation3D.hpp"
#include "stdint.h"

#define MPI_TIMED_BARRIER_TAG 110503

using rank_t = int;

using std::vector;

/*!
\brief Sends and revs data
\param tess The tessellation
\param cells The data to send/recv
\param ghost_or_sent True for ghost cells false for sent cells.
*/
template<class T>
void MPI_exchange_data(const Tessellation& tess, vector<T>& cells, bool ghost_or_sent);

template<class T>
void MPI_exchange_data(const Tessellation& tess, vector<T>& cells,bool ghost_or_sent)
{
	if (cells.empty())
		throw UniversalError("Empty cell vector in MPI_exchange_data");
	T example_cell = cells[0];
	vector<int> correspondents;
	vector<vector<int> > duplicated_points;
	if (ghost_or_sent)
	{
		correspondents = tess.GetDuplicatedProcs();
		duplicated_points = tess.GetDuplicatedPoints();
	}
	else
	{
		correspondents = tess.GetSentProcs();
		duplicated_points = tess.GetSentPoints();
	}
	vector<MPI_Request> req(correspondents.size());
	vector<vector<double> > tempsend(correspondents.size());
	vector<double> temprecv;
	double temp = 0;
	for (size_t i = 0; i < correspondents.size(); ++i)
	{
		bool isempty = duplicated_points[i].empty();
		if(!isempty)
			tempsend[i] = list_serialize(VectorValues(cells, duplicated_points[i]));
		int size = static_cast<int>(tempsend[i].size());
		if (size == 0)
			MPI_Isend(&temp, 1, MPI_DOUBLE, correspondents[i], 4, MPI_COMM_WORLD, &req[i]);
		else
			MPI_Isend(&tempsend[i][0], size, MPI_DOUBLE, correspondents[i], 5, MPI_COMM_WORLD, &req[i]);
	}
	const vector<vector<int> >& ghost_indices = tess.GetGhostIndeces();
	if (ghost_or_sent)
		cells.resize(tess.GetTotalPointNumber(),cells[0]);
	else
		cells = VectorValues(cells, tess.GetSelfPoint());
	vector<vector<T> > torecv(correspondents.size());
	for (size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_DOUBLE, &count);
		temprecv.resize(static_cast<size_t>(count));
		MPI_Recv(&temprecv[0], count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (status.MPI_TAG == 5)
		{
			size_t location = static_cast<size_t>(std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE) -
				correspondents.begin());
			if (location >= correspondents.size())
				throw UniversalError("Bad location in mpi exchange");
			torecv[location] = list_unserialize(temprecv, example_cell);
		}
		else
		{
			if (status.MPI_TAG != 4)
				throw UniversalError("Recv bad mpi tag (" + std::to_string(status.MPI_TAG) + ")");
		}
	}
	for (size_t i = 0; i < correspondents.size(); ++i)
	{
		if (ghost_or_sent)
		{
			for (size_t j = 0; j < torecv[i].size(); ++j)
				cells.at(ghost_indices.at(i).at(j)) = torecv[i][j];
		}
		else
		{
			for (size_t j = 0; j < torecv[i].size(); ++j)
				cells.push_back(torecv[i][j]);
		}
	}
	if (!req.empty())
		MPI_Waitall(static_cast<int>(correspondents.size()), &req[0], MPI_STATUSES_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
}

/*!
\brief Sends and revs data
\param tess The tessellation
\param cells The data to send/recv
\param ghost_or_sent True for ghost cells false for sent cells.
*/
template<class T>
void MPI_exchange_data(const Tessellation3D& tess, vector<T>& cells, bool ghost_or_sent,T const * example_cell =0);

template<class T>
void MPI_exchange_data(const Tessellation3D& tess, vector<T>& cells, bool ghost_or_sent, T const * example_cell)
{
	if (example_cell == 0 && cells.empty())
		throw UniversalError("Empty cell vector in MPI_exchange_data");
	if (example_cell == 0)
		example_cell = &cells[0];
	vector<int> correspondents;
	vector<vector<size_t> > duplicated_points;
	if (ghost_or_sent)
	{
		correspondents = tess.GetDuplicatedProcs();
		duplicated_points = tess.GetDuplicatedPoints();
	}
	else
	{
		correspondents = tess.GetSentProcs();
		duplicated_points = tess.GetSentPoints();
	}
	vector<MPI_Request> req(correspondents.size());
	vector<vector<double> > tempsend(correspondents.size());
	vector<double> temprecv;
	double temp = 0;
	for (size_t i = 0; i < correspondents.size(); ++i)
	{
		bool isempty = duplicated_points[i].empty();
		if (!isempty)
			tempsend[i] = list_serialize(VectorValues(cells, duplicated_points[i]));
		int size = static_cast<int>(tempsend[i].size());
		if (size == 0)
			MPI_Isend(&temp, 1, MPI_DOUBLE, correspondents[i], 4, MPI_COMM_WORLD, &req[i]);
		else
			MPI_Isend(&tempsend[i][0], size, MPI_DOUBLE, correspondents[i], 5, MPI_COMM_WORLD, &req[i]);
	}
	const vector<vector<size_t> >& ghost_indices = tess.GetGhostIndeces();
	if (ghost_or_sent)
		cells.resize(tess.GetTotalPointNumber(), *example_cell);
	else
		cells = VectorValues(cells, tess.GetSelfIndex());
	vector<vector<T> > torecv(correspondents.size());
	for (size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_DOUBLE, &count);
		temprecv.resize(static_cast<size_t>(count));
		MPI_Recv(&temprecv[0], count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (status.MPI_TAG == 5)
		{
			size_t location = static_cast<size_t>(std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE) -
				correspondents.begin());
			if (location >= correspondents.size())
				throw UniversalError("Bad location in mpi exchange");
			torecv[location] = list_unserialize(temprecv, *example_cell);
		}
		else
		{
			if (status.MPI_TAG != 4)
				throw UniversalError("Recv bad mpi tag (" + std::to_string(status.MPI_TAG) + ")");
		}
	}
	for (size_t i = 0; i < correspondents.size(); ++i)
	{
		if (ghost_or_sent)
		{
			for (size_t j = 0; j < torecv[i].size(); ++j)
				cells.at(ghost_indices.at(i).at(j)) = torecv[i][j];
		}
		else
		{
			for (size_t j = 0; j < torecv[i].size(); ++j)
				cells.push_back(torecv[i][j]);
		}
	}
	if (!req.empty())
		MPI_Waitall(static_cast<int>(correspondents.size()), &req[0], MPI_STATUSES_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
}

/*!
\brief Sends and revs data
\param totalkwith The cpus to talk with
\param tosend The indeces in data to send ordered by cpu
\param cells The data to send
\return The recv data ordered by cpu
*/
template <class T>
vector<vector<T> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<int> > const& tosend,
	vector<T>const& cells);

template <class T>
vector<vector<T> > MPI_exchange_data(const vector<int>& totalkwith,vector<vector<int> > const& tosend,
	vector<T>const& cells)
{
	assert(!cells.empty());
	vector<MPI_Request> req(totalkwith.size());
	vector<vector<double> > tempsend(totalkwith.size());
	vector<double> temprecv;
	double temp = 0;
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		bool isempty = tosend[i].empty();
		if (!isempty)
			tempsend[i] = list_serialize(VectorValues(cells, tosend[i]));
		int size = static_cast<int>(tempsend[i].size());
		if (size == 0)
			MPI_Isend(&temp, 1, MPI_DOUBLE, totalkwith[i], 4, MPI_COMM_WORLD, &req[i]);
		else
			MPI_Isend(&tempsend[i][0], size, MPI_DOUBLE, totalkwith[i], 5, MPI_COMM_WORLD, &req[i]);
	}
	vector<vector<T> > torecv(totalkwith.size());
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_DOUBLE, &count);
		temprecv.resize(static_cast<size_t>(count));
		MPI_Recv(&temprecv[0], count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (status.MPI_TAG == 5)
		{
			size_t location = static_cast<size_t>(std::find(totalkwith.begin(), totalkwith.end(), status.MPI_SOURCE) -
				totalkwith.begin());
			if (location >= totalkwith.size())
				throw UniversalError("Bad location in mpi exchange");
			torecv[location] = list_unserialize(temprecv, cells[0]);
		}
		else
		{
			if (status.MPI_TAG != 4)
				throw UniversalError("Recv bad mpi tag (" + std::to_string(status.MPI_TAG) + ")");
		}
	}
	if (!req.empty())
		MPI_Waitall(static_cast<int>(totalkwith.size()), &req[0], MPI_STATUSES_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
	return torecv;
}

/*!
\brief Sends and revs data
\param totalkwith The cpus to talk with
\param tosend The indeces in data to send ordered by cpu
\param cells The data to send
\return The recv data ordered by cpu
*/
template <class T>
vector<vector<T> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<size_t> > const& tosend,
	vector<T>const& cells);

template <class T>
vector<vector<T> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<size_t> > const& tosend,
	vector<T>const& cells)
{
	assert(!cells.empty());
	vector<MPI_Request> req(totalkwith.size());
	vector<vector<double> > tempsend(totalkwith.size());
	vector<double> temprecv;
	double temp = 0;
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		bool isempty = tosend[i].empty();
		if (!isempty)
			tempsend[i] = list_serialize(VectorValues(cells, tosend[i]));
		int size = static_cast<int>(tempsend[i].size());
		if (size == 0)
			MPI_Isend(&temp, 1, MPI_DOUBLE, totalkwith[i], 4, MPI_COMM_WORLD, &req[i]);
		else
			MPI_Isend(&tempsend[i][0], size, MPI_DOUBLE, totalkwith[i], 5, MPI_COMM_WORLD, &req[i]);
	} 
	vector<vector<T> > torecv(totalkwith.size());
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_DOUBLE, &count);
		temprecv.resize(static_cast<size_t>(count));
		MPI_Recv(&temprecv[0], count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (status.MPI_TAG == 5)
		{
			size_t location = static_cast<size_t>(std::find(totalkwith.begin(), totalkwith.end(), status.MPI_SOURCE) -
				totalkwith.begin());
			if (location >= totalkwith.size())
				throw UniversalError("Bad location in mpi exchange");
			torecv[location] = list_unserialize(temprecv, cells[0]);
		}
		else
		{
			if (status.MPI_TAG != 4)
				throw UniversalError("Recv bad mpi tag (" + std::to_string(status.MPI_TAG) + ")");
		}
	}
	if(!req.empty())
		MPI_Waitall(static_cast<int>(totalkwith.size()), &req[0], MPI_STATUSES_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
	return torecv;
}

template <class T>
vector<vector<T> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<T> > const& tosend, T const& demo);

template <class T>
vector<vector<T> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<T> > const& tosend,T const& demo)
{
	vector<MPI_Request> req(totalkwith.size());
	vector<vector<double> > tempsend(totalkwith.size());
	vector<double> temprecv;
	double temp = 0;
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		bool isempty = tosend[i].empty();
		if (!isempty)
			tempsend[i] = list_serialize(tosend[i]);
		int size = static_cast<int>(tempsend[i].size());
		if (size == 0)
			MPI_Isend(&temp, 1, MPI_DOUBLE, totalkwith[i], 4, MPI_COMM_WORLD, &req[i]);
		else
			MPI_Isend(&tempsend[i][0], size, MPI_DOUBLE, totalkwith[i], 5, MPI_COMM_WORLD, &req[i]);
	}
	vector<vector<T> > torecv(totalkwith.size());
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_DOUBLE, &count);
		temprecv.resize(static_cast<size_t>(count));
		MPI_Recv(&temprecv[0], count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (status.MPI_TAG == 5)
		{
			size_t location = static_cast<size_t>(std::find(totalkwith.begin(), totalkwith.end(), status.MPI_SOURCE) -
				totalkwith.begin());
			if (location >= totalkwith.size())
				throw UniversalError("Bad location in mpi exchange");
			torecv[location] = list_unserialize(temprecv, demo);
		}
		else
		{
			if (status.MPI_TAG != 4)
				throw UniversalError("Recv bad mpi tag (" + std::to_string(status.MPI_TAG) + ")");
		}
	}
	if (!req.empty())
		MPI_Waitall(static_cast<int>(totalkwith.size()), &req[0], MPI_STATUSES_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
	return torecv;
}

template <class T>
vector<vector<vector<T> > > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<vector<T > > > const& tosend,
	T const& demo);

template <class T>
vector<vector<vector<T> > > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<vector<T > > > const& tosend,
	T const& demo)
{
	vector<vector<vector<T > > > res(totalkwith.size());
	vector<MPI_Request> req(2*totalkwith.size());
	vector<vector<double> > tempsend(2*totalkwith.size());
	vector<vector<int> > send_sizes(totalkwith.size());
	//	vector<T> temprecv;
	double temp = 0;
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		bool isempty = tosend[i].empty();
		if (!isempty)
		{
			send_sizes[i].reserve(tosend[i].size());
			for (size_t j = 0; j < tosend[i].size(); ++j)
			{
				send_sizes[i].push_back(static_cast<int>(tosend[i][j].size()));
				vector<double> dtemp = list_serialize(tosend[i][j]);
				tempsend[i].insert(tempsend[i].end(), dtemp.begin(), dtemp.end());
			}		
		}
		else
			send_sizes[i].push_back(-1);
		int size = static_cast<int>(tempsend[i].size());
		if (size == 0)
		{
			MPI_Isend(&temp, 1, MPI_DOUBLE, totalkwith[i], 4, MPI_COMM_WORLD, &req[2 * i]);
			MPI_Isend(&send_sizes[i][0], 1, MPI_INT, totalkwith[i], 5, MPI_COMM_WORLD, &req[2 * i + 1]);
		}
		else
		{
			MPI_Isend(&tempsend[i][0], size, MPI_DOUBLE, totalkwith[i], 6, MPI_COMM_WORLD, &req[2 * i]);
			MPI_Isend(&send_sizes[i][0], static_cast<int>(send_sizes[i].size()), MPI_INT, totalkwith[i], 7, MPI_COMM_WORLD, &req[2 * i + 1]);
		}
	}
	vector<vector<double> > srecv(totalkwith.size());
	vector<vector<int> > irecv(totalkwith.size());
	for (size_t i = 0; i < 2 * totalkwith.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int count;
		if (status.MPI_TAG % 2 == 1)
		{
			MPI_Get_count(&status, MPI_INT, &count);
			size_t location = static_cast<size_t>(std::find(totalkwith.begin(), totalkwith.end(), status.MPI_SOURCE) -
				totalkwith.begin());
			if (location >= totalkwith.size())
				throw UniversalError("Bad location in mpi exchange");
			irecv[location].resize(static_cast<size_t>(count));
			MPI_Recv(&irecv[location][0], count, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else
		{
			MPI_Get_count(&status, MPI_DOUBLE, &count);
			size_t location = static_cast<size_t>(std::find(totalkwith.begin(), totalkwith.end(), status.MPI_SOURCE) -
				totalkwith.begin());
			if (location >= totalkwith.size())
				throw UniversalError("Bad location in mpi exchange");
			srecv[location].resize(static_cast<size_t>(count));
			MPI_Recv(&srecv[location][0], count, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	for (size_t i = 0; i < totalkwith.size(); ++i)
	{
		if (irecv[i].at(0) < 1)
			continue;
		vector<T> T_temp = list_unserialize(srecv[i], demo);
		if (T_temp.empty())
			continue;
		size_t counter = 0;
		for (size_t j = 0; j < irecv[i].size(); ++j)
		{
			if (irecv[i][j] < 1)
				continue;
			size_t size_add = static_cast<size_t>(irecv[i][j]);
			vector<T> T_add(T_temp.begin() + counter, T_temp.begin() + counter + size_add);
			res[i].push_back(T_add);
			counter += size_add;
		}
	}
	if (!req.empty())
		MPI_Waitall(static_cast<int>(2*totalkwith.size()), &req[0], MPI_STATUSES_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
	return res;
}

void MPI_exchange_data2(const Tessellation3D& tess, vector<double>& cells, bool ghost_or_sent);

vector<vector<double> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<double> > &tosend);

vector<vector<int> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<int> > &tosend);

vector<vector<vector<int> > > MPI_exchange_data(const Tessellation3D& tess, vector<vector<vector<int> > > &tosend);

vector<vector<vector<size_t> > > MPI_exchange_data(const Tessellation3D& tess, vector<vector<vector<size_t> > > &tosend);

vector<vector<vector<double> > > MPI_exchange_data(const Tessellation3D& tess, vector<vector<vector<double> > > &tosend);

vector<vector<size_t> > MPI_exchange_data(const vector<int>& totalkwith, vector<vector<size_t> > &tosend);

void MPI_exchange_data(const Tessellation3D& tess, vector<char>& cells, bool ghost_or_sent);

void MPI_Timed_barrier(const MPI_Comm &comm, double seconds, std::string const &place);

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_Spread(const Container<T, Ts...> &data, rank_t root, const MPI_Comm &comm)
{
	rank_t rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if(size == 1)
	{
		return data;
	}

	size_t totalSize;
	if(rank == root)
	{
		totalSize = data.size();
	}
	MPI_Bcast(&totalSize, 1, MPI_UNSIGNED_LONG, root, comm);

	if(totalSize == 0)
	{
		return std::vector<T>();
	}

	size_t elementSize;
	if(rank == root)
	{
		elementSize = data.at(0).getChunkSize();
	}
	MPI_Bcast(&elementSize, 1, MPI_UNSIGNED_LONG, root, comm);

	std::vector<double> myDataSerialized;
	if(rank == root)
	{
		for(const T &element : data)
		{
			std::vector<double> elementSerialized = element.serialize();
			myDataSerialized.insert(myDataSerialized.end(), elementSerialized.cbegin(), elementSerialized.cend());
		}
	}
	size_t idealSize = totalSize / size;
	size_t currentSize = 0;
	std::vector<int> displs, counts;
	for(rank_t _rank = 0; _rank < size; _rank++)
	{
		size_t _begin = _rank * idealSize;
		size_t _end = (_rank == size - 1)? totalSize : ((_rank + 1) * idealSize);
		size_t length = _end - _begin;
		counts.push_back(length * elementSize);
		displs.push_back(currentSize);
		currentSize += length * elementSize;
	}

	int myRecvNum = counts[rank];
	std::vector<double> receivedDataSerialized(myRecvNum);
	MPI_Scatterv(myDataSerialized.data(), counts.data(), displs.data(), MPI_DOUBLE, receivedDataSerialized.data(), myRecvNum, MPI_DOUBLE, root, comm);

	std::vector<T> toReturn;
	for(int i = 0; i < myRecvNum / elementSize; i++)
	{
		std::vector<double> elementData;
		for(size_t j = 0; j < elementSize; j++)
		{
			elementData.push_back(receivedDataSerialized[i * elementSize + j]);
		}
		toReturn.emplace_back();
		T &value = toReturn.back();
		value.unserialize(elementData);
	}
	return toReturn;
}

/**
 * More convenient all-to-all funciton, allowing to send serializable objects 
*/
template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Exchange_all_to_all(const std::vector<Container<T, Ts...>> &data, const MPI_Comm &comm)
{
	static_assert(is_serializable<T>::value, "MPI_Exchange_all_to_all: given type must be serializable");

	rank_t size;
	MPI_Comm_size(comm, &size);

	if(data.size() != static_cast<size_t>(size))
	{
		UniversalError eo("MPI_Exchange_all_to_all: data size must be equal to the number of ranks");
		eo.addEntry("Size", data.size());
		throw eo;
	}

	// agree on the chunk size
	size_t chunkSize = T().getChunkSize();
	if(chunkSize == 0)
	{
		return std::vector<std::vector<T>>(size); // no data is being sent
	}
	
	std::vector<int> sendDisplacements(size, 0);
	size_t totalSendSize = 0;
	std::vector<int> sizesToAll(size, 0), sizesFromAll(size, 0);
	for(rank_t _rank = 0; _rank < size; _rank++)
	{
		sizesToAll[_rank] = static_cast<int>(data[_rank].size()) * chunkSize;
		totalSendSize += sizesToAll[_rank];
		if(_rank > 0)
		{
			sendDisplacements[_rank ] = sendDisplacements[_rank - 1] + sizesToAll[_rank - 1];
		}
	}

	MPI_Alltoall(sizesToAll.data(), 1, MPI_INT, sizesFromAll.data(), 1, MPI_INT, comm);

	std::vector<double> allDataSend;
	allDataSend.reserve(totalSendSize);

	for(rank_t _rank = 0; _rank < size; _rank++)
	{
		for(const T &value : data[_rank])
		{
			std::vector<double> serialized = value.serialize();
			allDataSend.insert(allDataSend.end(), serialized.begin(), serialized.end());
		}
	}

	std::vector<int> recvDisplacements(size, 0);
	size_t totalRecvSize = 0;
	for(rank_t _rank = 0; _rank < size; _rank++)
	{
		totalRecvSize += sizesFromAll[_rank];
		if(_rank > 0)
		{
			recvDisplacements[_rank] = recvDisplacements[_rank - 1] + sizesFromAll[_rank - 1];
		}
	}

	std::vector<double> allDataRecv(totalRecvSize);
	
	MPI_Alltoallv(allDataSend.data(), sizesToAll.data(), sendDisplacements.data(), MPI_DOUBLE,
				  allDataRecv.data(), sizesFromAll.data(), recvDisplacements.data(), MPI_DOUBLE, comm);

	size_t i = 0;
	std::vector<std::vector<T>> resultByRanks(size);
	for(rank_t _rank = 0; _rank < size; _rank++)
	{
		// receive input
		size_t NumReceived = sizesFromAll[_rank] / chunkSize;
		resultByRanks[_rank].reserve(NumReceived);
		for(size_t j = 0; j < NumReceived; j++)
		{
			T value;
			value.unserialize(std::vector(allDataRecv.cbegin() + i, allDataRecv.cbegin() + i + chunkSize));
			resultByRanks[_rank].push_back(value);
			i += chunkSize;
		}
	}

	return resultByRanks;
}

template<typename T>
T MPI_Bcast_serializable(const T &data, rank_t root, const MPI_Comm &comm)
{
	static_assert(is_serializable<T>::value, "MPI_Exchange_all_to_all: given type must be serializable");

	rank_t rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	size_t chunkSize = data.getChunkSize();
	std::vector<double> recv(chunkSize);
	if(rank == root)
	{
		recv = data.serialize();
	}
	MPI_Bcast(recv.data(), chunkSize, MPI_DOUBLE, root, comm);
	T result;
	result.unserialize(recv);
	return result;
}

template<typename T>
std::vector<std::vector<T>> MPI_Exchange_by_ownership_by_ranks(const std::vector<T> &data, const std::function<rank_t(const T&)> &ownership, const MPI_Comm &comm)
{
	static_assert(is_serializable<T>::value, "MPI_Exchange_by_ownership_by_ranks: given type must be serializable");

	rank_t rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	std::vector<std::vector<T>> dataByRanks(size);
	for(const T &value : data)
	{
		rank_t _rank = ownership(value);
		if(_rank < 0 or _rank >= size)
		{
			UniversalError eo("MPI_Exchange_by_ownership_by_ranks: ownership function returned invalid rank");
			eo.addEntry("Rank", _rank);
			eo.addEntry("Size", size);
			throw eo;
		}
		dataByRanks[_rank].push_back(value);
	}

	return MPI_Exchange_all_to_all(dataByRanks, comm);
}

template<typename T>
std::vector<T> MPI_Exchange_by_ownership(const std::vector<T> &data, const std::function<rank_t(const T&)> &ownership, const MPI_Comm &comm)
{
	std::vector<std::vector<T>> exchangedData = MPI_Exchange_by_ownership_by_ranks(data, ownership, comm);
	std::vector<T> result;
	for(const std::vector<T> &values : exchangedData)
	{
		result.insert(result.end(), values.cbegin(), values.cend());
	}
	return result;
}

template<typename T>
std::vector<T> MPI_Gatherv_serializable(const std::vector<T> &data, rank_t root, const MPI_Comm &comm)
{
	static_assert(is_serializable<T>::value, "MPI_Exchange_by_ownership: given type must be serializable");

	rank_t rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// agree on the chunk size
	size_t chunkSize = T().getChunkSize();
	if(chunkSize == 0)
	{
		return std::vector<T>(); // no data is being sent
	}
	
	int mySendSize = static_cast<int>(data.size());
	std::vector<double> dataToSend;
	dataToSend.reserve(data.size() * chunkSize);

	for(const T &dataValue : data)
	{
		std::vector<double> serialized = dataValue.serialize();
		dataToSend.insert(dataToSend.end(), serialized.begin(), serialized.end());
	}

	if(rank == root)
	{
		std::vector<int> sizesToRecv(size), displs(size, 0);
		sizesToRecv.resize(size);

		MPI_Gather(&mySendSize, 1, MPI_INT, sizesToRecv.data(), 1, MPI_INT, root, comm);

		int allSize = 0;
		for(rank_t _rank = 0; _rank < size; _rank++)
		{
			displs[_rank] = allSize;
			sizesToRecv[_rank] *= chunkSize;
			allSize += sizesToRecv[_rank];
		}

		std::vector<double> dataToRecv(allSize);
		MPI_Gatherv(dataToSend.data(), dataToSend.size(), MPI_DOUBLE, dataToRecv.data(), sizesToRecv.data(), displs.data(), MPI_DOUBLE, root, comm);

		std::vector<T> received;
		std::vector<double>::const_iterator curr = dataToRecv.cbegin();
		for(int i = 0; i < allSize; i += chunkSize)
		{
			std::vector<double>::const_iterator _end = curr + chunkSize;
			received.emplace_back();
			received.back().unserialize(std::vector<double>(curr, _end));
			curr = _end;
		}
		return received;
	}
	else
	{
		MPI_Gather(&mySendSize, 1, MPI_INT, NULL, 0, MPI_INT, root, comm);
		MPI_Gatherv(dataToSend.data(), dataToSend.size(), MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, root, comm);
		return std::vector<T>();
	}
}


template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_All_cast_by_ranks(const Container<T, Ts...> &data, const MPI_Comm &comm)
{
	static_assert(is_serializable<T>::value, "MPI_All_cast_by_ranks: given type must be serializable");

	rank_t size;
	MPI_Comm_size(comm, &size);

	// agree the chunk size
	size_t chunkSize = 	T().getChunkSize();
	if(chunkSize == 0)
	{
		return std::vector<std::vector<T>>(size); // no data is being sent
	}

	// first know how much data is being sent from each one
	std::vector<double> dataToSend;
	for(const T &value : data)
	{
		std::vector<double> serialized = value.serialize();
		dataToSend.insert(dataToSend.end(), serialized.begin(), serialized.end());
	}
	int count = static_cast<int>(dataToSend.size());
	std::vector<int> recvCounts(size, 0);
	MPI_Allgather(&count, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);

	std::vector<int> recvDisplacements(size, 0);
	size_t totalToReceive = recvCounts[0];
	for(rank_t _rank = 1; _rank < size; _rank++)
	{
		recvDisplacements[_rank] = recvDisplacements[_rank - 1] + recvCounts[_rank - 1];
		totalToReceive += recvCounts[_rank];
	}
	std::vector<int> sendDisplacements(size, 0);
	std::vector<int> sendCounts(size, count);
	std::vector<double> dataToRecv(totalToReceive);
	MPI_Alltoallv(dataToSend.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE,
				  dataToRecv.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE, comm);

	std::vector<std::vector<T>> resultByRanks(size);
	for(rank_t _rank = 0; _rank < size; _rank++)
	{
		std::vector<T> &receiveFromRank = resultByRanks[_rank];
		size_t indexBegin = recvDisplacements[_rank];
		for(int i = 0; i < recvCounts[_rank]; i += chunkSize)
		{
			receiveFromRank.emplace_back(T());
			T &value = receiveFromRank.back();
			value.unserialize(std::vector(dataToRecv.cbegin() + indexBegin + i, dataToRecv.cbegin() + indexBegin + i + chunkSize));
		}
	}
	return resultByRanks;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_All_cast(const Container<T, Ts...> &data, const MPI_Comm &comm)
{
	static_assert(is_serializable<T>::value, "MPI_All_cast: given type must be serializable");

	std::vector<std::vector<T>> resultByRanks = MPI_All_cast_by_ranks(data, comm);
	std::vector<T> result;
	for(const std::vector<T> &values : resultByRanks)
	{
		result.insert(result.end(), values.cbegin(), values.cend());
	}
	return result;
}

#endif //RICH_MPI

#endif // MPI_COMMANDS_HPP

