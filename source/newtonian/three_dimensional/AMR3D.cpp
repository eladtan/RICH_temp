#include "AMR3D.hpp"
#include <boost/array.hpp>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <limits>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include "3D/tessellation/utils/PolyClip.hpp"
#include "misc/universal_error.hpp"
#include "misc/utils.hpp"
#include "newtonian/three_dimensional/SpatialReconstruction3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/cell_updater_3d.hpp"
#include "3D/tessellation/Neighbors.hpp"

//#define debug_amr 1

namespace
{
		const std::vector<Plane> &CachedPolyPlanes(Tessellation3D const& tess, size_t cell_index,
			std::unordered_map<size_t, std::vector<Plane> > &cache)
		{
			auto it = cache.find(cell_index);
			if(it != cache.end())
			{
				return it->second;
			}
			auto inserted = cache.emplace(cell_index, std::vector<Plane>());
			CreatePolyPlanes(tess, cell_index, inserted.first->second);
			return inserted.first->second;
		}

		const ClipBounds &CachedPolyBounds(Tessellation3D const& tess, size_t cell_index,
			std::unordered_map<size_t, ClipBounds> &cache)
		{
			auto it = cache.find(cell_index);
			if(it != cache.end())
			{
				return it->second;
			}
			auto inserted = cache.emplace(cell_index, CreatePolyBounds(tess, cell_index));
			return inserted.first->second;
		}

		void RemoveRefineNeighborRemove(Tessellation3D const& tess, std::vector<size_t> const& remove,
		std::vector<size_t> &refine, std::vector<Vector3D> &refine_direction)
	{
		vector<size_t> neigh;
		size_t Nrefine = refine.size();
		vector<size_t> newrefine;
		newrefine.reserve(Nrefine);
		std::vector<Vector3D> new_direction;
		new_direction.reserve(Nrefine);
		for (size_t i = 0; i < Nrefine; ++i)
		{
			tess.GetNeighbors(refine[i], neigh);
			neigh.push_back(refine[i]);
			bool good = true;
			for (size_t j = 0; j < neigh.size(); ++j)
			{
				if (std::binary_search(remove.begin(), remove.end(), neigh[j]))
				{
					good = false;
					break;
				}
			}
			if (good)
			{
				newrefine.push_back(refine[i]);
				if (!refine_direction.empty())
					new_direction.push_back(refine_direction[i]);
			}
		}
		refine = newrefine;
		refine_direction = new_direction;
	}

	std::pair<vector<size_t>, vector<double> > GreedyRemoveCandidates(
		vector<double> const& merits, vector<size_t> const& candidates,
		Tessellation3D const& tess)
	{
		if (merits.size() != candidates.size())
			throw UniversalError("Merits and Candidates don't have same size in GreedyRemoveCandidates");
		size_t N = candidates.size();

		vector<size_t> all_candidates = candidates;
		vector<double> all_merits = merits;

#ifdef RICH_MPI
		size_t Norg = tess.GetPointNo();
		// Exchange candidate info with MPI neighbor ranks
		vector<vector<size_t> > duplicated_indeces = tess.GetDuplicatedPoints();
		vector<vector<size_t> > sort_indeces(duplicated_indeces.size());
		for (size_t i = 0; i < duplicated_indeces.size(); ++i)
		{
			sort_index(duplicated_indeces[i], sort_indeces[i]);
			std::sort(duplicated_indeces[i].begin(), duplicated_indeces[i].end());
		}
		size_t nproc = duplicated_indeces.size();
		vector<vector<size_t> > indeces(nproc);
		vector<vector<double> > merit_send(nproc);
		vector<vector<size_t> > local_cells_sent(nproc);
		vector<size_t> neigh_tmp;
		for (size_t i = 0; i < N; ++i)
		{
			tess.GetNeighbors(candidates[i], neigh_tmp);
			bool has_ghost = false;
			for (size_t j = 0; j < neigh_tmp.size(); ++j)
			{
				if (neigh_tmp[j] >= Norg)
				{
					has_ghost = true;
					break;
				}
			}
			if (has_ghost)
			{
				for (size_t k = 0; k < nproc; ++k)
				{
					vector<size_t>::const_iterator it = binary_find(duplicated_indeces[k].begin(),
						duplicated_indeces[k].end(), candidates[i]);
					if (it != duplicated_indeces[k].end())
					{
						indeces[k].push_back(sort_indeces[k][static_cast<size_t>(it - duplicated_indeces[k].begin())]);
						merit_send[k].push_back(merits[i]);
						local_cells_sent[k].push_back(candidates[i]);
					}
				}
			}
		}

		// Build adjacency among candidates sent to the same remote rank.
		// The receiving rank cannot compute this because GetNeighbors is
		// unavailable for ghost cells (FacesInCell_ only covers local cells).
		vector<vector<size_t> > adj_send(nproc);
		for (size_t k = 0; k < nproc; ++k)
		{
			size_t M = local_cells_sent[k].size();
			if (M < 2)
				continue;
			std::unordered_map<size_t, size_t> cell_to_batch;
			for (size_t m = 0; m < M; ++m)
				cell_to_batch[local_cells_sent[k][m]] = m;
			for (size_t m = 0; m < M; ++m)
			{
				tess.GetNeighbors(local_cells_sent[k][m], neigh_tmp);
				for (size_t j = 0; j < neigh_tmp.size(); ++j)
				{
					std::unordered_map<size_t, size_t>::const_iterator it2 = cell_to_batch.find(neigh_tmp[j]);
					if (it2 != cell_to_batch.end() && it2->second > m)
					{
						adj_send[k].push_back(m);
						adj_send[k].push_back(it2->second);
					}
				}
			}
		}

		const std::vector<int> &amr_dupProcs = tess.GetDuplicatedProcs();
		indeces = MPI_exchange_data(amr_dupProcs, indeces);
		merit_send = MPI_exchange_data(amr_dupProcs, merit_send);
		adj_send = MPI_exchange_data(amr_dupProcs, adj_send);
		for (size_t i = 0; i < nproc; ++i)
		{
			if (!indeces[i].empty())
			{
				indeces[i] = VectorValues(tess.GetGhostIndeces()[i], indeces[i]);
				all_candidates.insert(all_candidates.end(), indeces[i].begin(), indeces[i].end());
				all_merits.insert(all_merits.end(), merit_send[i].begin(), merit_send[i].end());
			}
		}
#endif // RICH_MPI

		// Build candidate lookup set and adjacency map
		size_t Nall = all_candidates.size();
		std::unordered_set<size_t> candidate_set(all_candidates.begin(), all_candidates.end());

		// Build adjacency from local cells' GetNeighbors (captures local-local
		// and local-ghost edges; ghost-ghost edges are added below from exchange)
		std::unordered_map<size_t, vector<size_t> > candidate_adj;
		vector<size_t> neigh;
		for (size_t k = 0; k < Nall; ++k)
		{
			size_t cell = all_candidates[k];
#ifdef RICH_MPI
			if (cell >= Norg)
				continue;
#endif
			tess.GetNeighbors(cell, neigh);
			for (size_t j = 0; j < neigh.size(); ++j)
			{
				if (candidate_set.count(neigh[j]))
				{
					candidate_adj[cell].push_back(neigh[j]);
					if (neigh[j] != cell)
						candidate_adj[neigh[j]].push_back(cell);
				}
			}
		}

#ifdef RICH_MPI
		// Add ghost-ghost edges from the exchanged adjacency data
		for (size_t i = 0; i < nproc; ++i)
		{
			for (size_t p = 0; p + 1 < adj_send[i].size(); p += 2)
			{
				size_t ghost_a = indeces[i][adj_send[i][p]];
				size_t ghost_b = indeces[i][adj_send[i][p + 1]];
				candidate_adj[ghost_a].push_back(ghost_b);
				candidate_adj[ghost_b].push_back(ghost_a);
			}
		}
#endif

		// Sort by merit descending; break ties by cell index ascending
		vector<size_t> order(Nall);
		for (size_t i = 0; i < Nall; ++i)
			order[i] = i;
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
		{
			if (all_merits[a] != all_merits[b])
				return all_merits[a] > all_merits[b];
			return all_candidates[a] < all_candidates[b];
		});

		// Greedy: accept highest-merit candidate, exclude its candidate-neighbors
		std::unordered_set<size_t> excluded;
		vector<size_t> result_names;
		vector<double> result_merits;
		result_names.reserve(N);
		result_merits.reserve(N);
		for (size_t k = 0; k < Nall; ++k)
		{
			size_t idx = order[k];
			size_t cell = all_candidates[idx];
			if (excluded.count(cell))
				continue;
			// Accept this candidate (only add local cells to result)
#ifdef RICH_MPI
			if (cell < Norg)
			{
#endif
				result_names.push_back(cell);
				result_merits.push_back(all_merits[idx]);
#ifdef RICH_MPI
			}
#endif
			// Exclude all its candidate-neighbors
			std::unordered_map<size_t, vector<size_t> >::const_iterator adj_it = candidate_adj.find(cell);
			if (adj_it != candidate_adj.end())
			{
				for (size_t j = 0; j < adj_it->second.size(); ++j)
					excluded.insert(adj_it->second[j]);
			}
		}

		// Sort results by cell index for downstream compatibility
		vector<size_t> sort_idx = sort_index(result_names);
		result_names = VectorValues(result_names, sort_idx);
		result_merits = VectorValues(result_merits, sort_idx);
		return std::pair<vector<size_t>, vector<double> >(result_names, result_merits);
	}

	std::vector<Vector3D> GetNewPoints(Tessellation3D const& tess, std::pair<vector<size_t>,
		vector<Vector3D> > &ToRefine)
	{
#ifdef RICH_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		std::vector<size_t> bad_indeces;
		std::vector<size_t> neigh;
		size_t Nrefine = ToRefine.first.size();
		std::vector<Vector3D> res;
		res.reserve(Nrefine);
		// Do we have a prefred direction?
		if (!ToRefine.second.empty())
		{
			for (size_t i = 0; i < Nrefine; ++i)
			{
				double R = tess.GetWidth(ToRefine.first[i]);
				Vector3D newpoint = tess.GetMeshPoint(ToRefine.first[i]) + 0.25*R*ToRefine.second[i] / fastabs(ToRefine.second[i]);
				res.push_back(newpoint);
			}
		}
		else
		{
			// Find direction by opposite to closest point
			for (size_t i = 0; i < Nrefine; ++i)
			{
				// Find closest neighbor
				Vector3D point = tess.GetMeshPoint(ToRefine.first[i]);
				tess.GetNeighbors(ToRefine.first[i], neigh);
				double min_dist2 = ScalarProd(point - tess.GetMeshPoint(neigh[0]), point - tess.GetMeshPoint(neigh[0]));
				size_t index = 0;
				for (size_t j = 1; j < neigh.size(); ++j)
				{
					double temp = ScalarProd(point - tess.GetMeshPoint(neigh[j]), point - tess.GetMeshPoint(neigh[j]));
					if (temp < min_dist2)
					{
						index = j;
						min_dist2 = temp;
					}
				}
				// Make new point
				Vector3D n_split = (point - tess.GetMeshPoint(neigh[index]));
				n_split = normalize(n_split);
				res.push_back(point + 0.25*tess.GetWidth(ToRefine.first[i])*n_split);
			}
		}
		// Make sure point is inside domian
		std::pair<Vector3D, Vector3D> bb = tess.GetBoxCoordinates();
		for (size_t i = 0; i < Nrefine; ++i)
		{
			if (res[i].x > bb.second.x || res[i].x<bb.first.x || res[i].y>bb.second.y || res[i].y<bb.first.y
				|| res[i].z>bb.second.z || res[i].z < bb.first.z)
			{
				bad_indeces.push_back(i);
				continue;
			}
#ifdef RICH_MPI
			if (!tess.PointInMyDomain(res[i]))
			{
				// std::cout<<"Adding bad point "<<res[i]<<" index "<<i<<std::endl;
				bad_indeces.push_back(i);
				continue;
			}
#endif
		}
		if (!bad_indeces.empty())
		{
			RemoveVector(res, bad_indeces);
			RemoveVector(ToRefine.first, bad_indeces);
		}
		return res;
	}

#ifdef RICH_MPI
	void PackPolyhedron(const std::vector<Face> &poly, std::vector<double> &out)
	{
		out.clear();
		out.push_back(static_cast<double>(poly.size()));
		for (const Face &f : poly)
		{
			out.push_back(static_cast<double>(f.vertices.size()));
			for (const Vector3D &v : f.vertices)
			{
				out.push_back(v.x);
				out.push_back(v.y);
				out.push_back(v.z);
			}
		}
	}

	void UnpackPolyhedron(const double *data, size_t &offset, std::vector<Face> &poly)
	{
		size_t nfaces = static_cast<size_t>(data[offset++]);
		poly.resize(nfaces);
		for (size_t i = 0; i < nfaces; ++i)
		{
			size_t nverts = static_cast<size_t>(data[offset++]);
			poly[i].vertices.resize(nverts);
			for (size_t j = 0; j < nverts; ++j)
			{
				poly[i].vertices[j].x = data[offset++];
				poly[i].vertices[j].y = data[offset++];
				poly[i].vertices[j].z = data[offset++];
			}
		}
	}

	void PackPlanes(const std::vector<Plane> &planes, std::vector<double> &out)
	{
		out.clear();
		out.push_back(static_cast<double>(planes.size()));
		for (const Plane &p : planes)
		{
			out.push_back(ScalarProd(p.normal, p.point));
			out.push_back(p.normal.x);
			out.push_back(p.normal.y);
			out.push_back(p.normal.z);
		}
	}

	void UnpackPlanes(const double *data, size_t &offset, std::vector<Plane> &planes)
	{
		size_t nplanes = static_cast<size_t>(data[offset++]);
		planes.resize(nplanes);
		for (size_t i = 0; i < nplanes; ++i)
		{
			double d = data[offset++];
			planes[i].normal.x = data[offset++];
			planes[i].normal.y = data[offset++];
			planes[i].normal.z = data[offset++];
			if (std::abs(planes[i].normal.x) > 0.1)
				planes[i].point = Vector3D(d / planes[i].normal.x, 0.0, 0.0);
			else if (std::abs(planes[i].normal.y) > 0.1)
				planes[i].point = Vector3D(0.0, d / planes[i].normal.y, 0.0);
			else
				planes[i].point = Vector3D(0.0, 0.0, d / planes[i].normal.z);
		}
	}

	void PackBounds(const ClipBounds &b, std::vector<double> &out)
	{
		out.push_back(b.lower.x); out.push_back(b.lower.y); out.push_back(b.lower.z);
		out.push_back(b.upper.x); out.push_back(b.upper.y); out.push_back(b.upper.z);
		out.push_back(b.valid ? 1.0 : 0.0);
	}

	void UnpackBounds(const double *data, size_t &offset, ClipBounds &b)
	{
		b.lower.x = data[offset++]; b.lower.y = data[offset++]; b.lower.z = data[offset++];
		b.upper.x = data[offset++]; b.upper.y = data[offset++]; b.upper.z = data[offset++];
		b.valid = data[offset++] > 0.5;
	}

	struct ClipResultEntry
	{
		size_t task_id;
		double dv;
		Vector3D clip_CM;
	};

	void DistributeAndComputeClips(
		const std::vector<double> &packed_tasks,
		size_t local_task_count,
		const std::vector<double> &org_volumes,
		std::vector<ClipResultEntry> &results,
		MPI_Comm comm)
	{
		int myrank = 0, world_size = 1;
		MPI_Comm_rank(comm, &myrank);
		MPI_Comm_size(comm, &world_size);
		size_t ws = static_cast<size_t>(world_size);

		std::vector<int> task_counts(ws);
		int my_count = static_cast<int>(local_task_count);
		MPI_Allgather(&my_count, 1, MPI_INT, task_counts.data(), 1, MPI_INT, comm);

		int total_tasks = 0;
		for (size_t i = 0; i < ws; ++i)
			total_tasks += task_counts[i];

		int fair_share = (total_tasks + world_size - 1) / world_size;

		std::vector<std::vector<double>> send_data(ws);
		std::vector<std::vector<double>> send_volumes(ws);

		struct RankExcess { int rank; int excess; };
		struct RankDeficit { int rank; int deficit; };
		std::vector<RankExcess> overloaded;
		std::vector<RankDeficit> underloaded;
		for (int i = 0; i < world_size; ++i)
		{
			if (task_counts[i] > fair_share)
				overloaded.push_back({i, task_counts[i] - fair_share});
			else if (task_counts[i] < fair_share)
				underloaded.push_back({i, fair_share - task_counts[i]});
		}

		int my_send_count = 0;
		if (!underloaded.empty())
		{
			size_t u_idx = 0;
			int u_remaining = underloaded[0].deficit;

			struct Assignment { int target_rank; int count; };
			std::vector<Assignment> my_assignments;

			for (size_t o = 0; o < overloaded.size(); ++o)
			{
				int to_distribute = overloaded[o].excess;
				int distributed_so_far = 0;
				while (to_distribute > 0 && u_idx < underloaded.size())
				{
					int chunk = std::min(to_distribute, u_remaining);
					if (overloaded[o].rank == myrank)
						my_assignments.push_back({underloaded[u_idx].rank, chunk});
					to_distribute -= chunk;
					distributed_so_far += chunk;
					u_remaining -= chunk;
					if (u_remaining == 0)
					{
						u_idx++;
						if (u_idx < underloaded.size())
							u_remaining = underloaded[u_idx].deficit;
					}
				}
				if (overloaded[o].rank == myrank)
					my_send_count = distributed_so_far;
			}

			if (!my_assignments.empty())
			{
				size_t task_offset = 0;
				std::vector<size_t> task_offsets(local_task_count);
				for (size_t t = 0; t < local_task_count; ++t)
				{
					task_offsets[t] = task_offset;
					size_t nfaces = static_cast<size_t>(packed_tasks[task_offset]);
					task_offset++;
					for (size_t f = 0; f < nfaces; ++f)
					{
						size_t nverts = static_cast<size_t>(packed_tasks[task_offset]);
						task_offset++;
						task_offset += nverts * 3;
					}
					size_t nplanes = static_cast<size_t>(packed_tasks[task_offset]);
					task_offset++;
					task_offset += nplanes * 4;
					task_offset += 7; // source bounds
					task_offset += 7; // target bounds
				}

				size_t excess_start = static_cast<size_t>(fair_share);
				size_t sent = 0;
				for (const Assignment &a : my_assignments)
				{
					size_t tr = static_cast<size_t>(a.target_rank);
					for (int t = 0; t < a.count; ++t)
					{
						size_t task_idx = excess_start + sent;
						size_t start = task_offsets[task_idx];
						size_t end = (task_idx + 1 < local_task_count) ? task_offsets[task_idx + 1] : packed_tasks.size();
						send_data[tr].push_back(static_cast<double>(task_idx));
						send_data[tr].insert(send_data[tr].end(), packed_tasks.begin() + start, packed_tasks.begin() + end);
						send_volumes[tr].push_back(org_volumes[task_idx]);
						sent++;
					}
				}
			}
		}

		std::vector<std::vector<double>> recv_data = MPI_Exchange_all_to_all(send_data, comm);
		std::vector<std::vector<double>> recv_volumes = MPI_Exchange_all_to_all(send_volumes, comm);

		struct RemoteResult
		{
			int source_rank;
			size_t remote_task_id;
			double dv;
			Vector3D clip_CM;
		};
		std::vector<RemoteResult> remote_results;

		ClipWorkspace clip_workspace;
		std::vector<Face> poly;
		std::vector<Plane> planes;

		for (size_t r = 0; r < ws; ++r)
		{
			if (recv_data[r].empty())
				continue;
			size_t offset = 0;
			size_t vol_idx = 0;
			while (offset < recv_data[r].size())
			{
				size_t remote_task_id = static_cast<size_t>(recv_data[r][offset++]);
				UnpackPolyhedron(recv_data[r].data(), offset, poly);
				UnpackPlanes(recv_data[r].data(), offset, planes);
				ClipBounds source_bounds, target_bounds;
				UnpackBounds(recv_data[r].data(), offset, source_bounds);
				UnpackBounds(recv_data[r].data(), offset, target_bounds);
				double ov = recv_volumes[r][vol_idx++];

				auto [dv, clip_vol, clip_CM] = clipCells(poly, planes, clip_workspace, &source_bounds, &target_bounds);
				if (dv > ov * 1e-10)
				{
					RemoteResult rr;
					rr.source_rank = static_cast<int>(r);
					rr.remote_task_id = remote_task_id;
					rr.dv = dv;
					rr.clip_CM = clip_CM;
					remote_results.push_back(rr);
				}
			}
		}

		std::vector<std::vector<double>> result_send(ws);
		for (const RemoteResult &rr : remote_results)
		{
			result_send[rr.source_rank].push_back(static_cast<double>(rr.remote_task_id));
			result_send[rr.source_rank].push_back(rr.dv);
			result_send[rr.source_rank].push_back(rr.clip_CM.x);
			result_send[rr.source_rank].push_back(rr.clip_CM.y);
			result_send[rr.source_rank].push_back(rr.clip_CM.z);
		}
		std::vector<std::vector<double>> result_recv = MPI_Exchange_all_to_all(result_send, comm);

		results.clear();

		size_t sent_start = my_send_count > 0 ? static_cast<size_t>(fair_share) : local_task_count;
		size_t sent_end = sent_start + static_cast<size_t>(my_send_count);

		{
			size_t offset = 0;
			for (size_t t = 0; t < local_task_count; ++t)
			{
				UnpackPolyhedron(packed_tasks.data(), offset, poly);
				UnpackPlanes(packed_tasks.data(), offset, planes);
				ClipBounds source_bounds, target_bounds;
				UnpackBounds(packed_tasks.data(), offset, source_bounds);
				UnpackBounds(packed_tasks.data(), offset, target_bounds);
				if (t >= sent_start && t < sent_end)
					continue;
				auto [dv, clip_vol, clip_CM] = clipCells(poly, planes, clip_workspace, &source_bounds, &target_bounds);
				if (dv > org_volumes[t] * 1e-10)
				{
					ClipResultEntry e;
					e.task_id = t;
					e.dv = dv;
					e.clip_CM = clip_CM;
					results.push_back(e);
				}
			}
		}

		for (size_t r = 0; r < ws; ++r)
		{
			size_t offset = 0;
			while (offset + 5 <= result_recv[r].size())
			{
				ClipResultEntry e;
				e.task_id = static_cast<size_t>(result_recv[r][offset++]);
				e.dv = result_recv[r][offset++];
				e.clip_CM.x = result_recv[r][offset++];
				e.clip_CM.y = result_recv[r][offset++];
				e.clip_CM.z = result_recv[r][offset++];
				results.push_back(e);
			}
		}
	}
#endif // RICH_MPI

#ifdef RICH_MPI
	void SendRecvMPIFullRemove(Tessellation3D const& tess, vector<size_t> const& toremove, vector<vector<size_t> >
		&nghost_index, vector<vector<vector<size_t> > > &duplicated_index, vector<vector<vector<Vector3D> > > &planes,
		vector<vector<vector<double>>> &planes_d, SpatialReconstruction3D &interp)
	{
		vector<size_t> temp;
		size_t Nprocs = tess.GetDuplicatedProcs().size();
		nghost_index.clear();
		nghost_index.resize(Nprocs);
		duplicated_index.clear();
		duplicated_index.resize(Nprocs);
		planes_d.clear();
		planes_d.resize(Nprocs);
		planes.clear();
		planes.resize(Nprocs);
		vector<vector<size_t> > duplicated_points = tess.GetDuplicatedPoints();
		vector<vector<size_t> > ghost_points = tess.GetGhostIndeces();
		vector<vector<size_t> > sort_indeces(Nprocs), sort_indecesg(Nprocs);
		for (size_t i = 0; i < Nprocs; ++i)
		{
			sort_index(duplicated_points[i], sort_indeces[i]);
			sort(duplicated_points[i].begin(), duplicated_points[i].end());
			sort_index(ghost_points[i], sort_indecesg[i]);
			sort(ghost_points[i].begin(), ghost_points[i].end());
		}

		size_t nremove = toremove.size();
		size_t Norg = tess.GetPointNo();
		vector<Plane> r_planes;
		vector<vector<size_t> > ghosts(Nprocs);
		vector<Vector3D> v_plane;
		vector<double> d_plane;
		for (size_t i = 0; i < nremove; ++i)
		{
			for (auto &g : ghosts) g.clear();
			tess.GetNeighbors(toremove[i], temp);
			size_t Nneigh = temp.size();
			bool added = false;
			for (size_t j = 0; j < Nneigh; ++j)
			{
				if (temp[j] >= Norg)
				{
					for (size_t k = 0; k < Nprocs; ++k)
					{
						vector<size_t>::const_iterator it = binary_find(ghost_points[k].begin(),
							ghost_points[k].end(), temp[j]);
						if (it != ghost_points[k].end())
						{
							ghosts[k].push_back(sort_indecesg[k][static_cast<size_t>(it - ghost_points[k].begin())]);
							added = true;
						}
					}
				}
			}
			if (added)
			{
				CreatePolyPlanes(tess, toremove[i], r_planes);
				size_t nplanes = r_planes.size();
				v_plane.resize(nplanes);
				d_plane.resize(nplanes);
				for (size_t j = 0; j < nplanes; ++j)
				{
					d_plane[j] = ScalarProd(r_planes[j].normal, r_planes[j].point);
					v_plane[j].x = r_planes[j].normal.x;
					v_plane[j].y = r_planes[j].normal.y;
					v_plane[j].z = r_planes[j].normal.z;
				}
				for (size_t k = 0; k < Nprocs; ++k)
					if (!ghosts[k].empty())
					{
						duplicated_index[k].push_back(ghosts[k]);
						nghost_index[k].push_back(sort_indeces[k][static_cast<size_t>(binary_find(duplicated_points[k].begin(),
							duplicated_points[k].end(), toremove[i]) - duplicated_points[k].begin())]);
                        planes[k].push_back(v_plane);
                        planes_d[k].push_back(d_plane);
					}
			}
		}
		// send/recv the data
		const std::vector<int> &amr_dupProcs2 = tess.GetDuplicatedProcs();
		nghost_index = MPI_exchange_data(amr_dupProcs2, nghost_index);
		duplicated_index = MPI_exchange_data(tess, duplicated_index);
		planes = MPI_exchange_data(amr_dupProcs2, planes);
		planes_d = MPI_exchange_data(tess, planes_d);
		// convert the data
		for (size_t i = 0; i < Nprocs; ++i)
		{
			size_t size = nghost_index[i].size();
			assert(duplicated_index[i].size() == size);
			for (size_t j = 0; j < size; ++j)
			{
				nghost_index[i][j] = tess.GetGhostIndeces()[i].at(nghost_index[i][j]);
				size_t size2 = duplicated_index[i][j].size();
				for (size_t k = 0; k < size2; ++k)
					duplicated_index[i][j][k] = tess.GetDuplicatedPoints()[i].at(duplicated_index[i][j][k]);
			}
		}
	}

	void SendRecvMPIRefine(Tessellation3D const& tess,
		std::vector<size_t> const& refined_points,
		std::vector<size_t> const& ToRefine_for_send,
		Tessellation3D const& oldtess,
		PointsToNeighborsMap const& k_neighbors,
		std::vector<std::vector<std::vector<size_t> > > &neigh_index, std::vector<std::vector<double> > &planes,
		std::vector<std::vector<size_t> > &n_planes, std::vector<std::vector<size_t> > &changed_byouter)
	{
		int myrank = 0, world_size = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		size_t ws = static_cast<size_t>(world_size);

		assert(refined_points.size() == ToRefine_for_send.size());

		// Initialize world-rank-indexed structures (indexed by MPI rank, not duplicatedprocs index)
		neigh_index.clear();
		neigh_index.resize(ws);
		planes.clear();
		planes.resize(ws);
		n_planes.clear();
		n_planes.resize(ws);
		changed_byouter.clear();
		changed_byouter.resize(ws);

		vector<Plane> r_planes;
		boost::container::flat_map<int, std::vector<size_t> > rank_to_indices;

		for (size_t i = 0; i < refined_points.size(); ++i)
		{
			size_t old_cell = ToRefine_for_send[i];
			auto it = k_neighbors.find(old_cell);
			if (it == k_neighbors.end())
				continue;

			// Get the polyhedra planes of the new refined cell
			r_planes.clear();
			CreatePolyPlanes(tess, refined_points[i], r_planes);
			

			// Group remote neighbors by rank
			rank_to_indices.clear();
			for (const RemotePoint& rp : it->second)
			{
				if (rp.rank != myrank)
					rank_to_indices[rp.rank].push_back(rp.indexOnRank);
			}

			// Pack data for each remote rank
			for (auto& kv : rank_to_indices)
			{
				int target_rank = kv.first;
				auto& indices = kv.second;
				size_t tr = static_cast<size_t>(target_rank);

				changed_byouter[tr].push_back(refined_points[i]);
				neigh_index[tr].push_back(indices);
				size_t nplanes = r_planes.size();
				for (size_t j = 0; j < nplanes; ++j)
				{
					planes[tr].push_back(ScalarProd(r_planes[j].normal, r_planes[j].point));
					planes[tr].push_back(r_planes[j].normal.x);
					planes[tr].push_back(r_planes[j].normal.y);
					planes[tr].push_back(r_planes[j].normal.z);
				}
				n_planes[tr].push_back(nplanes);
			}
		}
		// Exchange using MPI_Exchange_all_to_all (communicates with all ranks, not just duplicatedprocs)
		neigh_index = MPI_Exchange_all_to_all(neigh_index, MPI_COMM_WORLD);
		planes = MPI_Exchange_all_to_all(planes, MPI_COMM_WORLD);
		n_planes = MPI_Exchange_all_to_all(n_planes, MPI_COMM_WORLD);
		// No post-exchange index conversion needed: indices from GetKOrderNeighbors
		// are already local cell indices on the receiving rank
	}	
#endif

	void LocalRemove(Tessellation3D const& oldtess, std::vector<size_t> const& ToRemove, AMRExtensiveUpdater3D const& eu,
			 std::vector<ComputationalCell3D> const& cells, EquationOfState const& eos,
		Tessellation3D const& tess, std::vector<Conserved3D> &extensives,SpatialReconstruction3D &interp)
	{
		std::vector<size_t> neigh,temp2;
		point_vec temp;
		std::vector<std::vector<int> > i_temp;
			size_t NRemove = ToRemove.size();
			size_t Norg = oldtess.GetPointNo();
			std::vector<Face> poly;
			ClipWorkspace clip_workspace;
			std::unordered_map<size_t, std::vector<Plane> > plane_cache;
			std::unordered_map<size_t, ClipBounds> bounds_cache;
			for (size_t i = 0; i < NRemove; ++i)
			{
			oldtess.GetNeighbors(ToRemove[i], neigh);
			double org_volume = oldtess.GetVolume(ToRemove[i]);
			size_t Nneigh = neigh.size();
			// Get old poly
				try
				{
					CreatePolyFaces(oldtess, ToRemove[i], poly);
					ClipBounds source_bounds = computeBounds(poly);
					for (size_t j = 0; j < Nneigh; ++j)
					{
					if (neigh[j] >= Norg)
						continue;
					
					size_t index_remove = static_cast<size_t>(std::lower_bound(ToRemove.begin(), ToRemove.end(), neigh[j])
						- ToRemove.begin());
					size_t target_index = neigh[j] - index_remove;
					const ClipBounds target_bounds = CachedPolyBounds(tess, target_index, bounds_cache);
					const std::vector<Plane> &target_planes = CachedPolyPlanes(tess, target_index, plane_cache);
				auto [dv, clip_vol, clip_CM] = clipCells(poly, target_planes, clip_workspace, &source_bounds, &target_bounds);
#ifdef RICH_DEBUG
					try
					{
#endif
						if(dv > org_volume * 1e-10)
						{
                            Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[ToRemove[i]], eos, dv, interp.GetSlopes()[ToRemove[i]],
                            oldtess.GetCellCM(ToRemove[i]), clip_CM);
							if(cells[ToRemove[i]].ID == -1)
							{
                                std::cout << "To add local remove" << toadd << std::endl;
                                std::cout << "volume " << dv << " org volume " << oldtess.GetVolume(ToRemove[i]) << std::endl;
							}
								extensives[target_index] += toadd;
								if(extensives[target_index].mass < 0)
								{
									// todo: print message
								}
						}
#ifdef RICH_DEBUG
					}
					catch (UniversalError &eo)
					{
						eo.addEntry("Error in LocalRemove", 0);
						eo.addEntry("Volume", dv.second[0]);
						eo.addEntry("Current remove", ToRemove[i]);
						eo.addEntry("Current remove ID", cells[ToRemove[i]].ID);
						eo.addEntry("New mass", extensives[neigh[j] - index_remove].mass);
						eo.addEntry("Remove index", i);
						eo.addEntry("neigh index",j);
						eo.addEntry("neigh", neigh[j]);
						eo.addEntry("index_remove", index_remove);
						throw;
					}
#endif

				}
			}
			catch(UniversalError &eo)
			{
				eo.addEntry("ID", cells[ToRemove[i]].ID);
				throw eo;
			}
			}
		}

#ifdef RICH_MPI
	void DistributedLocalRemove(Tessellation3D const& oldtess, std::vector<size_t> const& ToRemove,
		AMRExtensiveUpdater3D const& eu, std::vector<ComputationalCell3D> const& cells,
		EquationOfState const& eos, Tessellation3D const& tess,
		std::vector<Conserved3D> &extensives, SpatialReconstruction3D &interp)
	{
		size_t NRemove = ToRemove.size();
		size_t Norg = oldtess.GetPointNo();

		struct RemoveTaskInfo
		{
			size_t remove_idx;
			size_t target_index;
		};
		std::vector<RemoveTaskInfo> task_info;
		std::vector<double> packed_tasks;
		std::vector<double> org_volumes;

		std::vector<Face> poly;
		std::vector<size_t> neigh;
		std::vector<Plane> target_planes;
		std::vector<double> poly_packed, planes_packed;

		for (size_t i = 0; i < NRemove; ++i)
		{
			oldtess.GetNeighbors(ToRemove[i], neigh);
			double org_volume = oldtess.GetVolume(ToRemove[i]);
			CreatePolyFaces(oldtess, ToRemove[i], poly);
			ClipBounds source_bounds = computeBounds(poly);
			PackPolyhedron(poly, poly_packed);

			for (size_t j = 0; j < neigh.size(); ++j)
			{
				if (neigh[j] >= Norg)
					continue;
				size_t index_remove = static_cast<size_t>(std::lower_bound(ToRemove.begin(), ToRemove.end(), neigh[j])
					- ToRemove.begin());
				size_t target_index = neigh[j] - index_remove;

				CreatePolyPlanes(tess, target_index, target_planes);
				ClipBounds target_bounds = CreatePolyBounds(tess, target_index);
				PackPlanes(target_planes, planes_packed);

				packed_tasks.insert(packed_tasks.end(), poly_packed.begin(), poly_packed.end());
				packed_tasks.insert(packed_tasks.end(), planes_packed.begin(), planes_packed.end());
				PackBounds(source_bounds, packed_tasks);
				PackBounds(target_bounds, packed_tasks);

				org_volumes.push_back(org_volume);
				RemoveTaskInfo ti;
				ti.remove_idx = i;
				ti.target_index = target_index;
				task_info.push_back(ti);
			}
		}

		std::vector<ClipResultEntry> results;
		DistributeAndComputeClips(packed_tasks, task_info.size(), org_volumes, results, MPI_COMM_WORLD);

		for (const ClipResultEntry &e : results)
		{
			size_t ri = task_info[e.task_id].remove_idx;
			size_t target_index = task_info[e.task_id].target_index;
			Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[ToRemove[ri]], eos, e.dv,
				interp.GetSlopes()[ToRemove[ri]], oldtess.GetCellCM(ToRemove[ri]), e.clip_CM);
			extensives[target_index] += toadd;
		}
	}

	void DistributedMPIRemove(Tessellation3D const& oldtess, Tessellation3D const& tess,
		std::vector<size_t> const& ToRemove, AMRExtensiveUpdater3D const& eu,
		EquationOfState const& eos, std::vector<ComputationalCell3D> const& cells,
		std::vector<Conserved3D> &extensives, SpatialReconstruction3D &interp)
	{
		vector<vector<size_t> > nghost_index;
		vector<vector<vector<size_t> > > duplicate_index;
		vector<vector<vector<Vector3D> > > planes_v;
		vector<vector<vector<double> > > planes_d;
		SendRecvMPIFullRemove(oldtess, ToRemove, nghost_index, duplicate_index, planes_v, planes_d, interp);

		size_t Nproc = nghost_index.size();

		struct MpiRemoveTaskInfo
		{
			size_t proc_idx;
			size_t entry_idx;
			size_t local_cell_idx;
			size_t target_index;
		};
		std::vector<MpiRemoveTaskInfo> task_info;
		std::vector<double> packed_tasks;
		std::vector<double> org_volumes;
		std::vector<Plane> planes;
		std::vector<Face> poly;
		std::vector<double> poly_packed, planes_packed;

		for (size_t i = 0; i < Nproc; ++i)
		{
			for (size_t j = 0; j < duplicate_index[i].size(); ++j)
			{
				size_t Nplane = planes_v[i].at(j).size();
				planes.resize(Nplane);
				for (size_t k = 0; k < Nplane; ++k)
				{
					planes[k].normal = planes_v[i][j].at(k);
					if (std::abs(planes[k].normal.x) > 0.1)
						planes[k].point = Vector3D(planes_d[i][j][k] / planes[k].normal.x, 0.0, 0.0);
					else if (std::abs(planes[k].normal.y) > 0.1)
						planes[k].point = Vector3D(0.0, planes_d[i][j][k] / planes[k].normal.y, 0.0);
					else
						planes[k].point = Vector3D(0.0, 0.0, planes_d[i][j][k] / planes[k].normal.z);
				}
				PackPlanes(planes, planes_packed);

				for (size_t k = 0; k < duplicate_index[i][j].size(); ++k)
				{
					size_t index_remove = static_cast<size_t>(std::lower_bound(ToRemove.begin(), ToRemove.end(),
						duplicate_index[i][j].at(k)) - ToRemove.begin());
					size_t target_cell = duplicate_index[i][j][k] - index_remove;

					CreatePolyFaces(tess, target_cell, poly);
					ClipBounds source_bounds = computeBounds(poly);
					double org_volume = tess.GetVolume(target_cell);
					PackPolyhedron(poly, poly_packed);

					packed_tasks.insert(packed_tasks.end(), poly_packed.begin(), poly_packed.end());
					packed_tasks.insert(packed_tasks.end(), planes_packed.begin(), planes_packed.end());
					PackBounds(source_bounds, packed_tasks);
					ClipBounds dummy_target;
					PackBounds(dummy_target, packed_tasks);

					org_volumes.push_back(org_volume);
					MpiRemoveTaskInfo ti;
					ti.proc_idx = i;
					ti.entry_idx = j;
					ti.local_cell_idx = k;
					ti.target_index = target_cell;
					task_info.push_back(ti);
				}
			}
		}

		std::vector<ClipResultEntry> results;
		DistributeAndComputeClips(packed_tasks, task_info.size(), org_volumes, results, MPI_COMM_WORLD);

		for (const ClipResultEntry &e : results)
		{
			size_t pi = task_info[e.task_id].proc_idx;
			size_t ej = task_info[e.task_id].entry_idx;
			size_t target_cell = task_info[e.task_id].target_index;
			Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[nghost_index[pi][ej]], eos, e.dv,
				interp.GetSlopes()[nghost_index[pi][ej]], oldtess.GetCellCM(nghost_index[pi][ej]), e.clip_CM);
			extensives[target_cell] += toadd;
		}
	}
#endif

#ifdef RICH_MPI
	void MPIRemove(Tessellation3D const& oldtess, Tessellation3D const& tess, std::vector<size_t> const& ToRemove,
		AMRExtensiveUpdater3D const& eu, EquationOfState const& eos,
		std::vector<ComputationalCell3D> const& cells, std::vector<Conserved3D> &extensives,SpatialReconstruction3D &interp)
	{
		vector<vector<size_t> > nghost_index;
		vector<vector<vector<size_t> > > duplicate_index;
		vector<vector < vector<Vector3D> > > planes_v;
		vector<vector < vector<double> > > planes_d;
		std::vector<size_t>  temp2;
		point_vec temp;
		std::vector<std::vector<int> > i_temp;
		SendRecvMPIFullRemove(oldtess, ToRemove, nghost_index, duplicate_index, planes_v, planes_d, interp);
			size_t Nproc = nghost_index.size();
			std::vector<Plane> planes;
			std::vector<Face> poly;
			ClipWorkspace clip_workspace;
			for (size_t i = 0; i < Nproc; ++i)
		{
			size_t NremoveMPI = duplicate_index[i].size();
			for (size_t j = 0; j < NremoveMPI; ++j)
			{
				// build planes for outer point
				size_t Nplane = planes_v[i].at(j).size();
				planes.resize(Nplane);
				for (size_t k = 0; k < Nplane; ++k)
				{
					planes[k].normal = planes_v[i][j].at(k);
					if(std::abs(planes[k].normal.x) > 0.1)
					{
						planes[k].point = Vector3D(planes_d[i][j][k] / planes[k].normal.x, 0.0, 0.0);
					}
					else
					{
						if(std::abs(planes[k].normal.y) > 0.1)
						{
							planes[k].point = Vector3D(0.0, planes_d[i][j][k] / planes[k].normal.y, 0.0);
						}
						else
						{
							planes[k].point = Vector3D(0.0, 0.0, planes_d[i][j][k] / planes[k].normal.z);
						}
					}
				}
				size_t NChangeLocal = duplicate_index[i][j].size();
				for (size_t k = 0; k < NChangeLocal; ++k)
				{
					size_t index_remove = static_cast<size_t>(std::lower_bound(ToRemove.begin(), ToRemove.end(),
						duplicate_index[i][j].at(k)) - ToRemove.begin());
					try
						{
						CreatePolyFaces(tess, duplicate_index[i][j][k] - index_remove, poly);
						ClipBounds source_bounds = computeBounds(poly);
						double org_volume = tess.GetVolume(duplicate_index[i][j][k] - index_remove);
						{
						auto [dv, clip_vol, clip_CM] = clipCells(poly, planes, clip_workspace, &source_bounds);
						try
							{
								if(dv > org_volume * 1e-10)
								{
                                    Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[nghost_index[i][j]], eos, dv, interp.GetSlopes()[nghost_index[i][j]], oldtess.GetCellCM(nghost_index[i][j]),
                                                    clip_CM);
									extensives[duplicate_index[i][j][k] - index_remove] += toadd;
								}
							}
							catch(UniversalError &eo)
							{
								eo.addEntry("Error in MPIRemove", 0);
								eo.addEntry("Volume", dv);
								eo.addEntry("Current remove", nghost_index[i][j]);
								eo.addEntry("Current remove ID", cells[nghost_index[i][j]].ID);
								eo.addEntry("New mass", extensives[duplicate_index[i][j][k] - index_remove].mass);
								eo.addEntry("Remove index i", i);
								eo.addEntry("Remove index j", j);
								eo.addEntry("Remove index k", k);
								eo.addEntry("neigh index", j);
								eo.addEntry("duplicate_index[i][j][k] - index_remove", duplicate_index[i][j][k] - index_remove);
								eo.addEntry("index_remove", index_remove);
								throw;
							}
						}
					}
					catch(UniversalError &eo)
					{
						eo.addEntry("Error in MPIRemove", 0);
						eo.addEntry("Current remove", nghost_index[i][j]);
						eo.addEntry("Current remove ID", cells[nghost_index[i][j]].ID);
					}
					}
				}
			}
		}
#endif

	void LocalRefine(Tessellation3D const& oldtess, Tessellation3D const& tess, std::vector<size_t> const& ToRefine,
		std::vector<ComputationalCell3D> const& cells, EquationOfState const& eos, AMRExtensiveUpdater3D const&eu, std::vector<Conserved3D> &extensives,SpatialReconstruction3D &interp)
	{
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // RICH_MPI
		size_t Nrefine = ToRefine.size();
		std::vector<size_t> neigh, temp2;
		point_vec temp;
		std::vector<std::vector<int> > i_temp;
		// r3d_poly poly, poly2;
		boost::container::flat_set<size_t> checked;
		std::stack<size_t> tocheck;
			size_t Norg = tess.GetPointNo() - ToRefine.size();
			size_t Norg2 = oldtess.GetPointNo();
			std::vector<Face> polyhedron;
			ClipWorkspace clip_workspace;
			std::unordered_map<size_t, std::vector<Plane> > plane_cache;
			std::unordered_map<size_t, ClipBounds> bounds_cache;
			for (size_t i = 0; i < Nrefine; ++i)
			{
			checked.clear();
				// Get new cell poly
				CreatePolyFaces(tess, Norg + i, polyhedron);
				ClipBounds source_bounds = computeBounds(polyhedron);
				double org_volume = tess.GetVolume(Norg + i);
			bool print = false;
			if(print)
			{
				std::cout << "New poly is " << polyhedron << std::endl;
			}
			double new_volume = tess.GetVolume(Norg + i);
			double total_dv = 0, total_dv2 = 0;
			// Get neigh to check
			oldtess.GetNeighbors(ToRefine[i], neigh);
			size_t Nneigh = neigh.size();
			for (size_t j = 0; j < Nneigh; ++j)
				tocheck.push(neigh[j]);
			tocheck.push(ToRefine[i]);
			while (!tocheck.empty())
			{
				size_t cur_check = tocheck.top();
				tocheck.pop();
				// did we check this cell yet?
				if (checked.count(cur_check) == 1)
					continue;
				else // keep track of visted cells
					checked.insert(cur_check);
				if (cur_check >= Norg2)
					continue;
				const ClipBounds target_bounds = CachedPolyBounds(oldtess, cur_check, bounds_cache);
				const std::vector<Plane> &target_planes = CachedPolyPlanes(oldtess, cur_check, plane_cache);
			auto [dv, clip_vol, clip_CM] = clipCells(polyhedron, target_planes, clip_workspace, &source_bounds, &target_bounds, nullptr, cells[cur_check].ID == -1);
			if(cells[cur_check].ID == -1)
				{
					std::cout << "Clipping cell " << cur_check << " ID " << cells[cur_check].ID << " dv " << dv << " init volume " << oldtess.GetVolume(cur_check) << " refine index " << ToRefine[i] << std::endl;
				}
				if(dv > org_volume * 1e-10)
				{
					total_dv += dv;
					// Remove extensive from neigh cell and add to new cell
					// Remove extensive from neigh cell and add to new cell
#ifdef RICH_DEBUG
					try
					{
#endif
                        Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[cur_check], eos, dv, interp.GetSlopes()[cur_check],
                            oldtess.GetCellCM(cur_check), clip_CM);
						extensives[cur_check] -= toadd;
						//extensives[Norg2 + i].tracers.resize(toadd.tracers.size());
						extensives[Norg2 + i] += toadd;
						oldtess.GetNeighbors(cur_check, neigh);
						Nneigh = neigh.size();
						for (size_t j = 0; j < Nneigh; ++j)
							tocheck.push(neigh[j]);
#ifdef RICH_DEBUG
					}
					catch (UniversalError &eo)
					{
						eo.addEntry("Error in LocalRefine", 0);
						eo.addEntry("Volume", dv);
						eo.addEntry("Current check", cur_check);
						eo.addEntry("Current check ID",cells[cur_check].ID);
						eo.addEntry("Old mass", extensives[cur_check].mass);
						eo.addEntry("Old density", cells[cur_check].density);
						eo.addEntry("New mass", extensives[Norg + i].mass);
						eo.addEntry("Refine index", i);
						eo.addEntry("Norg", Norg2);
						throw;
					}
#endif
				}
			}
			if(total_dv < std::numeric_limits<double>::min() * 100)
			{
				extensives[Norg2 + i] = eu.ConvertPrimitveToExtensive3D(cells[ToRefine[i]], eos, tess.GetVolume(Norg + i), Slope3D(), 
					Vector3D(), Vector3D());
				extensives[ToRefine[i]] -= extensives[Norg2 + i];
				std::cout << "Warning no good poly localrefine loc "<<oldtess.GetMeshPoint(ToRefine[i])<<" volume "<<oldtess.GetVolume(ToRefine[i])<<" old ID "<<cells[ToRefine[i]].ID<<std::endl;
			}
			}
		}

#ifdef RICH_MPI
	void DistributedLocalRefine(Tessellation3D const& oldtess, Tessellation3D const& tess,
		std::vector<size_t> const& ToRefine, std::vector<ComputationalCell3D> const& cells,
		EquationOfState const& eos, AMRExtensiveUpdater3D const& eu,
		std::vector<Conserved3D> &extensives, SpatialReconstruction3D &interp)
	{
		size_t Nrefine = ToRefine.size();
		size_t Norg = tess.GetPointNo() - Nrefine;
		size_t Norg2 = oldtess.GetPointNo();

		struct TaskInfo
		{
			size_t refine_idx;
			size_t old_cell;
		};
		std::vector<TaskInfo> task_info;
		std::vector<double> packed_tasks;
		std::vector<double> org_volumes;

		std::vector<Face> polyhedron;
		std::vector<size_t> neigh;
		std::vector<Plane> target_planes;
		std::vector<double> poly_packed, planes_packed;

		for (size_t i = 0; i < Nrefine; ++i)
		{
			CreatePolyFaces(tess, Norg + i, polyhedron);
			ClipBounds source_bounds = computeBounds(polyhedron);
			double org_volume = tess.GetVolume(Norg + i);
			PackPolyhedron(polyhedron, poly_packed);

			boost::container::flat_set<size_t> ring;
			ring.insert(ToRefine[i]);
			oldtess.GetNeighbors(ToRefine[i], neigh);
			for (size_t j = 0; j < neigh.size(); ++j)
				if (neigh[j] < Norg2)
					ring.insert(neigh[j]);

			for (size_t old_cell : ring)
			{
				CreatePolyPlanes(oldtess, old_cell, target_planes);
				ClipBounds target_bounds = CreatePolyBounds(oldtess, old_cell);
				PackPlanes(target_planes, planes_packed);

				packed_tasks.insert(packed_tasks.end(), poly_packed.begin(), poly_packed.end());
				packed_tasks.insert(packed_tasks.end(), planes_packed.begin(), planes_packed.end());
				PackBounds(source_bounds, packed_tasks);
				PackBounds(target_bounds, packed_tasks);

				org_volumes.push_back(org_volume);
				TaskInfo ti;
				ti.refine_idx = i;
				ti.old_cell = old_cell;
				task_info.push_back(ti);
			}
		}

		std::vector<ClipResultEntry> results;
		DistributeAndComputeClips(packed_tasks, task_info.size(), org_volumes, results, MPI_COMM_WORLD);

		std::vector<double> total_dv(Nrefine, 0.0);
		// Per refined cell: track which old cells were already checked (the 1-ring)
		std::vector<boost::container::flat_set<size_t> > checked(Nrefine);
		std::vector<std::stack<size_t> > tocheck(Nrefine);
		for (size_t i = 0; i < Nrefine; ++i)
		{
			checked[i].insert(ToRefine[i]);
			oldtess.GetNeighbors(ToRefine[i], neigh);
			for (size_t j = 0; j < neigh.size(); ++j)
				if (neigh[j] < Norg2)
					checked[i].insert(neigh[j]);
		}

		for (const ClipResultEntry &e : results)
		{
			size_t ri = task_info[e.task_id].refine_idx;
			size_t old_cell = task_info[e.task_id].old_cell;
			total_dv[ri] += e.dv;
			Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[old_cell], eos, e.dv,
				interp.GetSlopes()[old_cell], oldtess.GetCellCM(old_cell), e.clip_CM);
			extensives[old_cell] -= toadd;
			extensives[Norg2 + ri] += toadd;
			// Seed expansion from this cell's neighbors
			oldtess.GetNeighbors(old_cell, neigh);
			for (size_t j = 0; j < neigh.size(); ++j)
				if (!checked[ri].count(neigh[j]))
					tocheck[ri].push(neigh[j]);
		}

		// Local flood-fill beyond the 1-ring for each refined cell
		ClipWorkspace clip_workspace;
		for (size_t i = 0; i < Nrefine; ++i)
		{
			if (tocheck[i].empty())
				continue;
			double org_volume = tess.GetVolume(Norg + i);
			CreatePolyFaces(tess, Norg + i, polyhedron);
			ClipBounds source_bounds = computeBounds(polyhedron);

			std::unordered_map<size_t, std::vector<Plane> > plane_cache;
			std::unordered_map<size_t, ClipBounds> bounds_cache;
			while (!tocheck[i].empty())
			{
				size_t cur_check = tocheck[i].top();
				tocheck[i].pop();
				if (checked[i].count(cur_check))
					continue;
				checked[i].insert(cur_check);
				if (cur_check >= Norg2)
					continue;

				const ClipBounds target_bounds = CachedPolyBounds(oldtess, cur_check, bounds_cache);
				const std::vector<Plane> &cur_planes = CachedPolyPlanes(oldtess, cur_check, plane_cache);
				auto [dv, clip_vol, clip_CM] = clipCells(polyhedron, cur_planes, clip_workspace,
					&source_bounds, &target_bounds, nullptr, false);

				if (dv > org_volume * 1e-10)
				{
					total_dv[i] += dv;
					Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[cur_check], eos, dv,
						interp.GetSlopes()[cur_check], oldtess.GetCellCM(cur_check), clip_CM);
					extensives[cur_check] -= toadd;
					extensives[Norg2 + i] += toadd;
					oldtess.GetNeighbors(cur_check, neigh);
					for (size_t j = 0; j < neigh.size(); ++j)
						tocheck[i].push(neigh[j]);
				}
			}
		}

		for (size_t i = 0; i < Nrefine; ++i)
		{
			if (total_dv[i] < std::numeric_limits<double>::min() * 100)
			{
				extensives[Norg2 + i] = eu.ConvertPrimitveToExtensive3D(cells[ToRefine[i]], eos,
					tess.GetVolume(Norg + i), Slope3D(), Vector3D(), Vector3D());
				extensives[ToRefine[i]] -= extensives[Norg2 + i];
				std::cout << "Warning no good poly localrefine loc " << oldtess.GetMeshPoint(ToRefine[i])
					<< " volume " << oldtess.GetVolume(ToRefine[i]) << " old ID " << cells[ToRefine[i]].ID << std::endl;
			}
		}
	}

	void DistributedMPIRefine(Tessellation3D const& oldtess, Tessellation3D const& tess,
		std::vector<size_t> const& ToRefine, AMRExtensiveUpdater3D const& eu,
		EquationOfState const& eos, std::vector<ComputationalCell3D> const& cells,
		std::vector<Conserved3D> &extensives, SpatialReconstruction3D &interp)
	{
		int myrank = 0, world_size = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		size_t ws = static_cast<size_t>(world_size);

		std::vector<size_t> temp;
		size_t Norg = oldtess.GetPointNo();
		size_t Nrefine = ToRefine.size();

		std::vector<size_t> boundary_refine_indices;
		std::vector<size_t> neigh_neigh;
		for (size_t i = 0; i < Nrefine; ++i)
		{
			oldtess.GetNeighbors(ToRefine[i], temp);
			bool has_ghost_k2 = false;
			for (size_t j = 0; j < temp.size(); ++j)
			{
				if (temp[j] >= Norg && !oldtess.IsPointOutsideBox(temp[j]))
				{
					has_ghost_k2 = true;
					break;
				}
				if (temp[j] < Norg)
				{
					oldtess.GetNeighbors(temp[j], neigh_neigh);
					for (size_t k = 0; k < neigh_neigh.size(); ++k)
					{
						if (neigh_neigh[k] >= Norg && !oldtess.IsPointOutsideBox(neigh_neigh[k]))
						{
							has_ghost_k2 = true;
							break;
						}
					}
					if (has_ghost_k2)
						break;
				}
			}
			if (has_ghost_k2)
				boundary_refine_indices.push_back(i);
		}

		std::vector<size_t> boundary_cells;
		boundary_cells.reserve(boundary_refine_indices.size());
		for (size_t idx : boundary_refine_indices)
			boundary_cells.push_back(ToRefine[idx]);

		PointsToNeighborsMap k2_neighbors = GetKOrderNeighbors(oldtess, boundary_cells, 2, true, MPI_COMM_WORLD);

		std::vector<size_t> refined_points, ToRefine_for_send;
		for (size_t idx : boundary_refine_indices)
		{
			auto it = k2_neighbors.find(ToRefine[idx]);
			if (it != k2_neighbors.end())
			{
				bool has_remote = false;
				for (const RemotePoint& rp : it->second)
					if (rp.rank != myrank) { has_remote = true; break; }
				if (has_remote)
				{
					refined_points.push_back(tess.GetPointNo() - Nrefine + idx);
					ToRefine_for_send.push_back(ToRefine[idx]);
				}
			}
		}

		std::vector<std::vector<size_t> > n_planes;
		std::vector<std::vector<std::vector<size_t> > > neigh_index;
		std::vector<std::vector<double> > planes_flat;
		std::vector<std::vector<size_t> > changed_byouter;
		SendRecvMPIRefine(tess, refined_points, ToRefine_for_send, oldtess, k2_neighbors,
			neigh_index, planes_flat, n_planes, changed_byouter);

		struct MpiRefineTaskInfo
		{
			size_t rank_idx;
			size_t entry_idx;
			size_t local_cell;
		};
		std::vector<MpiRefineTaskInfo> task_info;
		std::vector<double> packed_tasks;
		std::vector<double> org_volumes;
		std::vector<std::vector<Conserved3D> > extensive_tosend(ws);

		std::vector<Plane> r_planes;
		std::vector<Face> polyhedron;
		std::vector<double> poly_packed, planes_packed;

		for (size_t i = 0; i < ws; ++i)
		{
			extensive_tosend[i].resize(neigh_index[i].size());
			size_t counter = 0;
			for (size_t j = 0; j < neigh_index[i].size(); ++j)
			{
				r_planes.resize(n_planes[i][j]);
				for (size_t k = 0; k < n_planes[i][j]; ++k)
				{
					r_planes[k].normal.x = planes_flat[i][counter + 1];
					r_planes[k].normal.y = planes_flat[i][counter + 2];
					r_planes[k].normal.z = planes_flat[i].at(counter + 3);
					if (std::abs(r_planes[k].normal.x) > 0.1)
						r_planes[k].point = Vector3D(planes_flat[i][counter] / r_planes[k].normal.x, 0.0, 0.0);
					else if (std::abs(r_planes[k].normal.y) > 0.1)
						r_planes[k].point = Vector3D(0.0, planes_flat[i][counter] / r_planes[k].normal.y, 0.0);
					else
						r_planes[k].point = Vector3D(0.0, 0.0, planes_flat[i][counter] / r_planes[k].normal.z);
					counter += 4;
				}
				PackPlanes(r_planes, planes_packed);

				for (size_t ki = 0; ki < neigh_index[i][j].size(); ++ki)
				{
					size_t local_cell = neigh_index[i][j][ki];
					if (local_cell >= Norg) continue;
					CreatePolyFaces(oldtess, local_cell, polyhedron);
					ClipBounds source_bounds = computeBounds(polyhedron);
					double org_volume = oldtess.GetVolume(local_cell);
					PackPolyhedron(polyhedron, poly_packed);

					packed_tasks.insert(packed_tasks.end(), poly_packed.begin(), poly_packed.end());
					packed_tasks.insert(packed_tasks.end(), planes_packed.begin(), planes_packed.end());
					PackBounds(source_bounds, packed_tasks);
					ClipBounds dummy;
					PackBounds(dummy, packed_tasks);

					org_volumes.push_back(org_volume);
					MpiRefineTaskInfo ti;
					ti.rank_idx = i;
					ti.entry_idx = j;
					ti.local_cell = local_cell;
					task_info.push_back(ti);
				}
			}
		}

		std::vector<ClipResultEntry> results;
		DistributeAndComputeClips(packed_tasks, task_info.size(), org_volumes, results, MPI_COMM_WORLD);

		for (const ClipResultEntry &e : results)
		{
			size_t ri = task_info[e.task_id].rank_idx;
			size_t ej = task_info[e.task_id].entry_idx;
			size_t local_cell = task_info[e.task_id].local_cell;
			Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[local_cell], eos, e.dv,
				interp.GetSlopes()[local_cell], oldtess.GetCellCM(local_cell), e.clip_CM);
			extensives[local_cell] -= toadd;
			extensive_tosend[ri][ej] += toadd;
		}

		extensive_tosend = MPI_Exchange_all_to_all(extensive_tosend, MPI_COMM_WORLD);
		size_t Nremove = oldtess.GetPointNo() + Nrefine - tess.GetPointNo();
		for (size_t i = 0; i < ws; ++i)
			for (size_t j = 0; j < extensive_tosend[i].size(); ++j)
				extensives[changed_byouter[i][j] + Nremove] += extensive_tosend[i][j];
	}
#endif

	#ifdef RICH_MPI
	void MPIRefine(Tessellation3D const& oldtess, Tessellation3D const& tess, std::vector<size_t> const& ToRefine,
		AMRExtensiveUpdater3D const& eu, EquationOfState const& eos, 
		std::vector<ComputationalCell3D> const& cells, std::vector<Conserved3D> &extensives,SpatialReconstruction3D &interp)
	{
		int myrank = 0, world_size = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		size_t ws = static_cast<size_t>(world_size);

		std::vector<size_t> temp, temp2;
		point_vec ptemp;
		size_t Norg = oldtess.GetPointNo();
		size_t Nrefine = ToRefine.size();

		std::vector<size_t> boundary_refine_indices;
		std::vector<size_t> neigh_neigh;
		for (size_t i = 0; i < Nrefine; ++i)
		{
			oldtess.GetNeighbors(ToRefine[i], temp);
			bool has_ghost_k2 = false;
			for (size_t j = 0; j < temp.size(); ++j)
			{
				if (temp[j] >= Norg && !oldtess.IsPointOutsideBox(temp[j]))
				{
					has_ghost_k2 = true;
					break;
				}
				if (temp[j] < Norg)
				{
					oldtess.GetNeighbors(temp[j], neigh_neigh);
					for (size_t k = 0; k < neigh_neigh.size(); ++k)
					{
						if (neigh_neigh[k] >= Norg && !oldtess.IsPointOutsideBox(neigh_neigh[k]))
						{
							has_ghost_k2 = true;
							break;
						}
					}
					if (has_ghost_k2)
						break;
				}
			}
			if (has_ghost_k2)
				boundary_refine_indices.push_back(i);
		}

		std::vector<size_t> boundary_cells;
		boundary_cells.reserve(boundary_refine_indices.size());
		for (size_t idx : boundary_refine_indices)
			boundary_cells.push_back(ToRefine[idx]);

		PointsToNeighborsMap k2_neighbors = GetKOrderNeighbors(oldtess, boundary_cells, 2, true, MPI_COMM_WORLD);

		std::vector<size_t> refined_points;
		std::vector<size_t> ToRefine_for_send;
		for (size_t idx : boundary_refine_indices)
		{
			auto it = k2_neighbors.find(ToRefine[idx]);
			if (it != k2_neighbors.end())
			{
				bool has_remote = false;
				for (const RemotePoint& rp : it->second)
				{
					if (rp.rank != myrank)
					{
						has_remote = true;
						break;
					}
				}
				if (has_remote)
				{
					refined_points.push_back(tess.GetPointNo() - Nrefine + idx);
					ToRefine_for_send.push_back(ToRefine[idx]);
				}
			}
		}

		std::vector<std::vector<size_t> > n_planes;
		std::vector<std::vector<std::vector<size_t> > > neigh_index;
		std::vector < std::vector<double> > planes;
		std::vector<std::vector<size_t> > changed_byouter;
		SendRecvMPIRefine(tess, refined_points, ToRefine_for_send, oldtess, k2_neighbors,
			neigh_index, planes, n_planes, changed_byouter);

		// Step 5: Find the intersections
		std::vector<Plane> r_planes;
		std::vector<std::vector<int> > i_temp;
		boost::container::flat_set<size_t> checked;
		std::stack<size_t> tocheck;
		std::vector<std::vector<Conserved3D> > extensive_tosend(ws);
		int rank = 0;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			std::vector<Face> polyhedron;
			ClipWorkspace clip_workspace;
			std::unordered_map<size_t, ClipBounds> bounds_cache;
			for (size_t i = 0; i < ws; ++i)
		{
			extensive_tosend[i].resize(neigh_index[i].size());
			size_t counter = 0;
			for (size_t j = 0; j < neigh_index[i].size(); ++j)
			{
				checked.clear();
				// Create planes for intersection
				r_planes.resize(n_planes[i][j]);
				for (size_t k = 0; k < n_planes[i][j]; ++k)
				{
					r_planes[k].normal.x = planes[i][counter + 1];
					r_planes[k].normal.y = planes[i][counter + 2];
					r_planes[k].normal.z = planes[i].at(counter + 3);
					if(std::abs(r_planes[k].normal.x) > 0.1)
					{
						r_planes[k].point = Vector3D(planes[i][counter] / r_planes[k].normal.x, 0.0, 0.0);
					}
					else
					{
						if(std::abs(r_planes[k].normal.y) > 0.1)
						{
							r_planes[k].point = Vector3D(0.0, planes[i][counter] / r_planes[k].normal.y, 0.0);
						}
						else
						{
							r_planes[k].point = Vector3D(0.0, 0.0, planes[i][counter] / r_planes[k].normal.z);
						}
					}
					counter += 4;
				}
				// Check for intersections
				for (size_t k = 0; k < neigh_index[i][j].size(); ++k)
					tocheck.push(neigh_index[i][j][k]);
				while (!tocheck.empty())
				{
					size_t cur_check = tocheck.top();
					tocheck.pop();
					// did we check this cell yet?
					if (checked.count(cur_check) == 1)
						continue;
					else // keep track of visted cells
						checked.insert(cur_check);
					if (cur_check >= Norg)
						continue;
					CreatePolyFaces(oldtess, cur_check, polyhedron);
					const ClipBounds source_bounds = CachedPolyBounds(oldtess, cur_check, bounds_cache);
					double org_volume = oldtess.GetVolume(cur_check);
				auto [dv, clip_vol, clip_CM] = clipCells(polyhedron, r_planes, clip_workspace, &source_bounds);

				if(dv > org_volume * 1e-10)
					{
							// add and remove the extensive
#ifdef RICH_DEBUG
						try
						{
#endif
						Conserved3D toadd = eu.ConvertPrimitveToExtensive3D(cells[cur_check], eos, dv, interp.GetSlopes()[cur_check],
							oldtess.GetCellCM(cur_check), clip_CM);
							extensives[cur_check] -= toadd;
							extensive_tosend[i][j] += toadd;
							oldtess.GetNeighbors(cur_check, temp);
							size_t Nneigh = temp.size();
							for (size_t k = 0; k < Nneigh; ++k)
								tocheck.push(temp[k]);
#ifdef RICH_DEBUG
						}
						catch (UniversalError &eo)
						{
							eo.addEntry("Error in MPIRefine", 0);
							eo.addEntry("Volume", dv);
							eo.addEntry("Current check", cur_check);
							eo.addEntry("Current check ID", cells[cur_check].ID);
							eo.addEntry("Old mass", extensives[cur_check].mass);
							eo.addEntry("Old density", cells[cur_check].density);
							eo.addEntry("Refine index i", i);
							eo.addEntry("Refine index j", j);
							throw;
						}
#endif
					}
				}
			}
		}
		extensive_tosend = MPI_Exchange_all_to_all(extensive_tosend, MPI_COMM_WORLD);
		size_t Nremove = oldtess.GetPointNo() + Nrefine - tess.GetPointNo();
			for (size_t i = 0; i < ws; ++i)
			{
				for (size_t j = 0; j < extensive_tosend[i].size(); ++j)
					extensives[changed_byouter[i][j] + Nremove] += extensive_tosend[i][j];
			}
		}
#endif
}

AMRCellUpdater3D::AMRCellUpdater3D(void) = default;

AMRCellUpdater3D::~AMRCellUpdater3D(void) = default;

AMRCellUpdater3D::AMRCellUpdater3D(const AMRCellUpdater3D&) = default;

AMRCellUpdater3D& AMRCellUpdater3D::operator=(const AMRCellUpdater3D&) = default;

AMRExtensiveUpdater3D::AMRExtensiveUpdater3D(const AMRExtensiveUpdater3D&) = default;

AMRExtensiveUpdater3D& AMRExtensiveUpdater3D::operator=
(const AMRExtensiveUpdater3D&) = default;

AMRExtensiveUpdater3D::AMRExtensiveUpdater3D(void) = default;

AMRExtensiveUpdater3D::~AMRExtensiveUpdater3D(void) = default;

Conserved3D SimpleAMRExtensiveUpdater3D::ConvertPrimitveToExtensive3D(const ComputationalCell3D& cell, const EquationOfState& eos,
	double volume, Slope3D const& slope, Vector3D const& CMold, Vector3D const& CMnew) const
{
	Conserved3D res;
	Vector3D diff(CMnew);
	diff -= CMold;
	
	ComputationalCell3D cell_temp(cell);
	ComputationalCellAddMult(cell_temp, slope.xderivative, diff.x);
	ComputationalCellAddMult(cell_temp, slope.yderivative, diff.y);
	ComputationalCellAddMult(cell_temp, slope.zderivative, diff.z);
	const double mass = volume * cell_temp.density;
	res.mass = mass;
	res.momentum = mass * cell.velocity + volume * cell.density * (cell_temp.velocity - cell.velocity);
	res.internal_energy = cell_temp.internal_energy * mass;
	res.energy = res.internal_energy + 0.5 * mass * ScalarProd(cell.velocity, cell.velocity) + volume * cell.density * ScalarProd(cell.velocity, cell_temp.velocity - cell.velocity);
	res.Erad = mass * cell_temp.Erad;
	res.Erad_dt = cell.Erad_dt * mass;
	res.Erad_dt_dt = cell.Erad_dt_dt * mass;
	size_t N = cell_temp.tracers.size();
	//res.tracers.resize(N);
	for (size_t i = 0; i < N; ++i)
		res.tracers[i] = cell_temp.tracers[i] * mass;
	for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
		res.Eg[g] = cell_temp.Eg[g] * mass;
	return res;
}

Conserved3D SimpleAMRExtensiveUpdaterSR3D::ConvertPrimitveToExtensive3D(const ComputationalCell3D& cell, const EquationOfState& eos,
	double volume, Slope3D const& slope, Vector3D const& CMold, Vector3D const& CMnew) const
{
	Conserved3D res;
	Vector3D diff(CMnew);
	diff -= CMold;
	ComputationalCell3D cell_temp(cell);
	ComputationalCellAddMult(cell_temp, slope.xderivative, diff.x);
	ComputationalCellAddMult(cell_temp, slope.yderivative, diff.y);
	ComputationalCellAddMult(cell_temp, slope.zderivative, diff.z);
	cell_temp.internal_energy = eos.dp2e(cell_temp.density, cell_temp.pressure, cell_temp.tracers, ComputationalCell3D::tracerNames);
	double gamma = 1.0 / std::sqrt(1 - ScalarProd(cell_temp.velocity, cell_temp.velocity));
	const double mass = volume * cell_temp.density*gamma;
	res.mass = mass;
	size_t N = cell_temp.tracers.size();
	//res.tracers.resize(N);
	for (size_t i = 0; i < N; ++i)
		res.tracers[i] = cell_temp.tracers[i] * mass;
	double enthalpy = cell_temp.internal_energy;
	if (fastabs(cell_temp.velocity) < 1e-5)
		res.energy = (gamma*enthalpy + 0.5*ScalarProd(cell_temp.velocity, cell_temp.velocity))* mass - cell_temp.pressure*volume;
	else
		res.energy = (gamma*enthalpy + (gamma - 1))* mass - cell_temp.pressure*volume;
	res.momentum = mass * cell_temp.velocity *gamma*(enthalpy + 1);
	return res;
}

SimpleAMRCellUpdater3D::SimpleAMRCellUpdater3D(const vector<string>& toskip) :toskip_(toskip) {}

ComputationalCell3D SimpleAMRCellUpdater3D::ConvertExtensiveToPrimitve3D(Conserved3D& extensive, const EquationOfState& eos,
	double volume, ComputationalCell3D const& old_cell) const
{
	for (size_t i = 0; i < toskip_.size(); ++i)
	{
	  if (*safe_retrieve(old_cell.stickers.begin(), ComputationalCell3D::stickerNames.begin(), ComputationalCell3D::stickerNames.end(), toskip_[i]))
			return old_cell;
	}

	ComputationalCell3D res;
	const double vol_inv = 1.0 / volume;
	res.density = extensive.mass*vol_inv;
	res.velocity = extensive.momentum / extensive.mass;
	res.ID  = old_cell.ID;
	size_t N = extensive.tracers.size();
	for (size_t i = 0; i < N; ++i)
		res.tracers[i] = extensive.tracers[i] / extensive.mass;
	res.stickers = old_cell.stickers;
	auto entropy_it = binary_find(ComputationalCell3D::tracerNames.begin(), ComputationalCell3D::tracerNames.end(), "Entropy");
	try
	{
		if(entropy_it != ComputationalCell3D::tracerNames.end())
		{
			size_t const entropy_index = static_cast<size_t>(entropy_it - ComputationalCell3D::tracerNames.begin());
			double new_energy = std::max(extensive.internal_energy / extensive.mass, eos.dp2e(res.density, eos.sd2p(res.tracers[entropy_index], res.density)));
			res.internal_energy = new_energy;
			extensive.internal_energy = new_energy * extensive.mass;
			extensive.energy = extensive.internal_energy + 0.5 * ScalarProd(res.velocity, res.velocity) * extensive.mass;
			res.pressure = eos.de2p(res.density, res.internal_energy);
			res.tracers[entropy_index] = eos.dp2s(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
			extensive.tracers[entropy_index] = res.tracers[entropy_index] * extensive.mass;
		}
		else
		{
			extensive.internal_energy = extensive.energy - 0.5 * ScalarProd(extensive.momentum, extensive.momentum) / extensive.mass;
			res.internal_energy = extensive.internal_energy / extensive.mass;
			res.pressure = eos.de2p(res.density, res.internal_energy);
		}
	}
	catch (UniversalError &eo)
	{
		eo.addEntry("Density", res.density);
		eo.addEntry("Vx", res.velocity.x);
		eo.addEntry("Vy", res.velocity.y);
		eo.addEntry("Vz", res.velocity.z);
		eo.addEntry("ID", static_cast<double>(res.ID));
		eo.addEntry("internal energy", extensive.internal_energy / extensive.mass);
		eo.addEntry("Volume", 1.0 / vol_inv);
		throw;
	}
	res.Erad = extensive.Erad / extensive.mass;
	res.temperature = eos.de2T(res.density, res.internal_energy);
	for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
		res.Eg[g] = extensive.Eg[g] / extensive.mass;
	res.Erad_dt = extensive.Erad_dt / extensive.mass;
	res.Erad_dt_dt = extensive.Erad_dt_dt / extensive.mass;
	return res;
}

SimpleAMRCellUpdaterSR3D::SimpleAMRCellUpdaterSR3D(double G, const vector<string>& toskip) : G_(G), toskip_(toskip) {}

ComputationalCell3D SimpleAMRCellUpdaterSR3D::ConvertExtensiveToPrimitve3D(Conserved3D& extensive, const EquationOfState& /*eos*/,
	double volume, ComputationalCell3D const& old_cell) const
{
	for (size_t i = 0; i < toskip_.size(); ++i)
	  if (*safe_retrieve(old_cell.stickers.begin(), ComputationalCell3D::stickerNames.begin(), ComputationalCell3D::stickerNames.end(), toskip_[i]))
			return old_cell;
		//if (safe_retrieve(old_cell.stickers, tracerstickernames.sticker_names, toskip_[i]))
			//return old_cell;

	double v = 0; // TODO: GetVelocity(extensive, G_) — undefined, needs implementation
	volume = 1.0 / volume;
	ComputationalCell3D res;
	if (res.density < 0)
		throw UniversalError("Negative density in SimpleAMRCellUpdaterSR");
	res.velocity = (fastabs(extensive.momentum)*1e8 < extensive.mass) ? extensive.momentum / extensive.mass : v * extensive.momentum / abs(extensive.momentum);
	double gamma_1 = std::sqrt(1 - ScalarProd(res.velocity, res.velocity));
	res.density = extensive.mass *gamma_1*volume;
	res.stickers = old_cell.stickers;
	//	size_t N = extensive.tracers.size();
//	res.tracers.resize(N);
	for (size_t i = 0; i < extensive.tracers.size(); ++i)
		res.tracers[i] = extensive.tracers[i] / extensive.mass;
	if (fastabs(res.velocity) < 1e-5)
		res.pressure = (G_ - 1)*((extensive.energy - ScalarProd(extensive.momentum, res.velocity))*volume
			+ (0.5*ScalarProd(res.velocity, res.velocity))*res.density);
	else
		res.pressure = (G_ - 1)*(extensive.energy*volume - ScalarProd(extensive.momentum, res.velocity)*volume
			+ (1.0 / gamma_1 - 1)*res.density);
	return res;
}

SimpleAMRExtensiveUpdater3D::SimpleAMRExtensiveUpdater3D(void) {}

CellsToRemove3D::~CellsToRemove3D(void) {}

CellsToRefine3D::~CellsToRefine3D(void) {}

AMR3D::AMR3D(EquationOfState const& eos, 
	     CellsToRefine3D const& refine,
	     CellsToRemove3D const& remove,
	     SpatialReconstruction3D &interp,
	     AMRCellUpdater3D* cu,
	     AMRExtensiveUpdater3D* eu,
	     bool distribute_clips):
  eos_(eos), 
  refine_(refine), 
  remove_(remove),
  scu_(SimpleAMRCellUpdater3D()),
  seu_(SimpleAMRExtensiveUpdater3D()), 
  interp_(interp), 
  cu_(cu), 
  eu_(eu),
  distribute_clips_(distribute_clips)
{
	if (!cu)
		cu_ = &scu_;
	if (!eu)
		eu_ = &seu_;
}


void AMR3D::operator() (Simulation &sim)
{
	int rank = 0;
#ifdef RICH_MPI
	int ws = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	Tessellation3D &tess = sim.getTessellation();
	std::vector<ComputationalCell3D> &cells = sim.getCells();
	std::vector<Conserved3D> &extensives = sim.getExtensives();
	EquationOfState const& eos = eos_;
	double time = sim.GetTime();
	// Get remove list
	std::pair<vector<size_t>, vector<double> > ToRemove = remove_.ToRemove(tess, cells, time);
	// sort
	vector<size_t> indeces = sort_index(ToRemove.first);
	ToRemove.second = VectorValues(ToRemove.second, indeces);
	ToRemove.first = VectorValues(ToRemove.first, indeces);
	// Remove neighboring candidates via greedy independent set
	ToRemove = GreedyRemoveCandidates(ToRemove.second, ToRemove.first, tess);
	// Get points to refine
	std::pair<vector<size_t>, std::vector<Vector3D> > ToRefine = refine_.ToRefine(tess, cells, time);
	sort_index(ToRefine.first, indeces);
	sort(ToRefine.first.begin(), ToRefine.first.end());
	if (!ToRefine.second.empty())
		VectorValues(ToRefine.second, indeces);
	// remove neighboring refine/remove
	RemoveRefineNeighborRemove(tess, ToRemove.first, ToRefine.first, ToRefine.second);

	// Do we need to rebuild tess ?
	int nAMR[2];
	nAMR[0] = static_cast<int>(ToRemove.first.size());
	nAMR[1] = static_cast<int>(ToRefine.first.size());
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, nAMR, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
	if ((nAMR[0] + nAMR[1]) == 0)
		return;
	if(rank == 0)
		std::cout<<"Removing "<<nAMR[0]<<" cells and refining "<<nAMR[1]<<" cells."<<std::endl;

	interp_.BuildSlopes(tess, cells, time);
	// Get new points from refine
	std::vector<Vector3D> new_points = GetNewPoints(tess, ToRefine);
	// Create copy of old tess
	boost::scoped_ptr<Tessellation3D> oldtess(tess.clone());
	// Build new tess
	vector<Vector3D> new_mesh = tess.getMeshPoints();
	size_t Norg = tess.GetPointNo();
	new_mesh.resize(Norg);
	RemoveVector(new_mesh, ToRemove.first);
	new_mesh.insert(new_mesh.end(), new_points.begin(), new_points.end());
#ifdef RICH_MPI
	std::vector<size_t> mask(Norg);
	std::iota(mask.begin(), mask.end(), 0);
	RemoveVector(mask, ToRemove.first);
	mask.resize(new_mesh.size(), std::numeric_limits<size_t>::max());
	tess.PreparePoints(new_mesh, mask);
	tess.BuildParallel(new_mesh, true /* no rebalance */, true /* no exchange */);
#else // RICH_MPI
	tess.Build(new_mesh);
#endif
	// Fix extensives for refine
	extensives.resize(oldtess->GetPointNo() + ToRefine.first.size());
#ifdef RICH_MPI
	if (distribute_clips_)
	{
		DistributedLocalRefine(*oldtess, tess, ToRefine.first, cells, eos, *eu_, extensives, interp_);
		DistributedMPIRefine(*oldtess, tess, ToRefine.first, *eu_, eos, cells, extensives, interp_);
	}
	else
	{
		LocalRefine(*oldtess, tess, ToRefine.first, cells, eos, *eu_, extensives, interp_);
		MPIRefine(*oldtess, tess, ToRefine.first, *eu_, eos, cells, extensives, interp_);
	}
#else
	LocalRefine(*oldtess, tess, ToRefine.first, cells, eos, *eu_, extensives, interp_);
#endif
	// Remove from extensive the remove cells
	RemoveVector(extensives, ToRemove.first);
	// Add the removed extensive to the neighboring cells
#ifdef RICH_MPI
	if (distribute_clips_)
	{
		DistributedLocalRemove(*oldtess, ToRemove.first, *eu_, cells, eos_, tess, extensives, interp_);
		DistributedMPIRemove(*oldtess, tess, ToRemove.first, *eu_, eos, cells, extensives, interp_);
	}
	else
	{
		LocalRemove(*oldtess, ToRemove.first, *eu_, cells, eos_, tess, extensives, interp_);
		MPIRemove(*oldtess, tess, ToRemove.first, *eu_, eos, cells, extensives, interp_);
	}
#else
	LocalRemove(*oldtess, ToRemove.first, *eu_, cells, eos_, tess, extensives, interp_);
#endif
	// Recalc cells
	RemoveVector(cells, ToRemove.first);
	size_t NorgNew = tess.GetPointNo();
	cells.resize(NorgNew);
	for (size_t i = 0; i < (Norg - ToRemove.first.size()); ++i)
	{
		try
		{
			cells[i] = cu_->ConvertExtensiveToPrimitve3D(extensives[i], eos, tess.GetVolume(i), cells[i]);
		}
		catch (UniversalError & eo)
		{
			eo.addEntry("First loop", static_cast<double>(i));
			eo.addEntry("Norg", static_cast<double>(Norg));
			throw;
		}
	}

	// Get index for ID
	size_t Nrefine = ToRefine.first.size();
	size_t Nstart = sim.GetMaxID() + 1;
#ifdef RICH_MPI
	std::vector<size_t> nrecv(static_cast<size_t>(ws), 0);
	MPI_Allgather(&Nrefine, 1, MPI_UNSIGNED_LONG_LONG, &nrecv[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
	for (size_t i = 0; i < static_cast<size_t>(rank); ++i)
		Nstart += nrecv[i];
#endif
	// Add new refined cells	
	for (size_t i = 0; i< ToRefine.first.size(); ++i)
	{
		size_t index_remove = static_cast<size_t>(std::lower_bound(ToRemove.first.begin(), ToRemove.first.end(),
			ToRefine.first[i]) - ToRemove.first.begin());
		try
		{
			cells[(Norg - ToRemove.first.size()) + i] = cu_->ConvertExtensiveToPrimitve3D(extensives[(Norg -
				ToRemove.first.size()) + i], eos, tess.GetVolume((Norg - ToRemove.first.size()) + i),
				cells[ToRefine.first[i] - index_remove]);
			// Add new ID
			cells[(Norg - ToRemove.first.size()) + i].ID = Nstart + i;
		}
		catch (UniversalError & eo)
		{
			eo.addEntry("Second loop", static_cast<double>(i));
			eo.addEntry("Norg", static_cast<double>(Norg));
			eo.addEntry("Nrefine", static_cast<double>(ToRefine.first.size()));
			throw;
		}
	}
	// Recalc entropy if needed
	size_t entropy_index = static_cast<size_t>(std::find(ComputationalCell3D::tracerNames.begin(), ComputationalCell3D::tracerNames.end(), std::string("Entropy")) - ComputationalCell3D::tracerNames.begin());
	if (entropy_index < ComputationalCell3D::tracerNames.size())
	{
		size_t Nentropy = cells.size();
		for (size_t i = 0; i < Nentropy; ++i)
		{
		  cells[i].tracers[entropy_index] = eos.dp2s(cells[i].density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
			extensives[i].tracers[entropy_index] = cells[i].tracers[entropy_index] * extensives[i].mass;
		}
	}
	size_t & MaxID = sim.GetMaxID();
#ifdef RICH_MPI
	// Update cells
	MPI_exchange_data(tess, cells, true);
	//#endif
	// Update Max ID
	for (size_t i = 0; i < static_cast<size_t>(ws); ++i)
		MaxID += nrecv[i];
#else
	MaxID += Nrefine;
#endif
}

AMR3D::~AMR3D(void) {}
