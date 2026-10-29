#include "Ghost3D.hpp"
#include "misc/utils.hpp"

namespace
{
	struct GhostSource
	{
		size_t source;	// index into the boundary face list: this ghost's largest face
		size_t other;	// its largest other face, or the list size when the ghost has a single face
	};

	// One entry per ghost point, in increasing ghost index. A ghost point mirrors one real point (its source) across a box
	// face. The mirror symmetry also makes degenerate faces (a point, a neighbor and their two mirrors are concyclic), so a
	// neighbor of the source can touch the same ghost through a face of ~zero area (2.3e-18 next to 2e-5 in the 2D_h1 wedge
	// collision, 2026-10-22); such a face carries no flux, and the ghost takes the state behind its largest face. Inserting
	// one entry per face instead skipped the duplicate and left the insertion hint past the end (a crash in FreeFlow).
	vector<GhostSource> GetGhostSources(Tessellation3D const& tess, vector<std::pair<size_t, size_t> > const& ghosts,
		vector<size_t> const& ghost_indices)
	{
		size_t const N = ghosts.size();
		vector<size_t> order;
		sort_index(ghost_indices, order);
		vector<GhostSource> res;
		res.reserve(N);
		for (size_t first = 0; first < N;)
		{
			size_t last = first + 1;
			while (last < N && ghost_indices[order[last]] == ghost_indices[order[first]])
				++last;
			GhostSource entry{order[first], N};
			for (size_t k = first + 1; k < last; ++k)
			{
				size_t const face = order[k];
				if (tess.GetArea(ghosts[face].first) > tess.GetArea(ghosts[entry.source].first))
				{
					entry.other = entry.source;
					entry.source = face;
				}
				else if (entry.other == N || tess.GetArea(ghosts[face].first) > tess.GetArea(ghosts[entry.other].first))
					entry.other = face;
			}
			res.push_back(entry);
			first = last;
		}
		return res;
	}
}

Ghost3D::~Ghost3D(void) {}

vector<std::pair<size_t, size_t> > Ghost3D::GetOuterFacesIndeces(Tessellation3D const& tess)const
{
	size_t N = tess.GetTotalFacesNumber();
	size_t Npoints = tess.GetPointNo();
	vector<std::pair<size_t, size_t> > res;
	for (size_t i = 0; i < N; ++i)
	{
		if (tess.BoundaryFace(i))
		{
			if(tess.GetFaceNeighbors(i).first>=Npoints)
				res.push_back(std::pair<size_t, size_t>(i, 1));
			else
				res.push_back(std::pair<size_t, size_t>(i, 2));
		}
	}
	return res;
}


void RigidWallGenerator3D::operator()(const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells, double /*time*/, boost::container::flat_map<size_t, ComputationalCell3D> &res) const
{
	vector<std::pair<size_t, size_t> > ghosts = GetOuterFacesIndeces(tess);
	size_t N = ghosts.size();
	vector<std::pair<size_t, ComputationalCell3D> > temp(N);
	vector<size_t> indeces(N);
	for (size_t i = 0; i < N; ++i)
	{
		size_t ghost_index = 0, real_index = 0;
		if (ghosts[i].second == 1)
		{
			ghost_index = tess.GetFaceNeighbors(ghosts[i].first).first;
			real_index = tess.GetFaceNeighbors(ghosts[i].first).second;
		}
		else
		{
			ghost_index = tess.GetFaceNeighbors(ghosts[i].first).second;
			real_index = tess.GetFaceNeighbors(ghosts[i].first).first;
		}
		Vector3D normal = normalize(tess.GetMeshPoint(ghost_index) - tess.GetMeshPoint(real_index));
		temp[i] = std::pair<size_t, ComputationalCell3D>(ghost_index, cells[real_index]);
		temp[i].second.velocity -= 2*normal*ScalarProd(normal, temp[i].second.velocity);
		indeces[i] = ghost_index;
	}
	vector<GhostSource> const sources = GetGhostSources(tess, ghosts, indeces);
	res.clear();
	res.reserve(sources.size());
	for (GhostSource const& entry : sources)
	{
		// Two substantial faces to one ghost point are a real inconsistency
		if (entry.other != N && tess.GetArea(ghosts[entry.other].first) > 1e-6 * tess.GetArea(ghosts[entry.source].first))
		{
			size_t const i = entry.other;
			UniversalError eo("bad insert in rigidghost");
			if(ghosts[i].second == 1)
				eo.addEntry("ID", cells[tess.GetFaceNeighbors(ghosts[i].first).second].ID);
			else
				eo.addEntry("ID", cells[tess.GetFaceNeighbors(ghosts[i].first).first].ID);
			// The ghost point is the outside neighbor of more than one substantial boundary face: list those faces
			size_t const duplicate = temp[i].first;
			Vector3D const ghost_point = tess.GetMeshPoint(duplicate);
			eo.addEntry("ghost index", static_cast<double>(duplicate));
			eo.addEntry("ghost x", ghost_point.x);
			eo.addEntry("ghost y", ghost_point.y);
			eo.addEntry("ghost z", ghost_point.z);
			for (size_t j = 0; j < N; ++j)
			{
				if (temp[j].first != duplicate)
					continue;
				size_t const face = ghosts[j].first;
				size_t const real = ghosts[j].second == 1 ? tess.GetFaceNeighbors(face).second : tess.GetFaceNeighbors(face).first;
				Vector3D const real_point = tess.GetMeshPoint(real);
				eo.addEntry("face", static_cast<double>(face));
				eo.addEntry("face area", tess.GetArea(face));
				eo.addEntry("real ID", static_cast<double>(cells[real].ID));
				eo.addEntry("real x", real_point.x);
				eo.addEntry("real y", real_point.y);
				eo.addEntry("real z", real_point.z);
				eo.addEntry("real volume", tess.GetVolume(real));
			}
			throw eo;
		}
		res.insert(res.end(), temp[entry.source]);
	}
}

Slope3D RigidWallGenerator3D::GetGhostGradient(const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& /*cells*/,
	const vector<Slope3D>& /*gradients*/, size_t /*ghost_index*/, double /*time*/, size_t /*face_index*/) const
{
	Slope3D res;
//	res.xderivative.tracers.resize(cells[0].tracers.size(), 0);
//	res.yderivative.tracers.resize(cells[0].tracers.size(), 0);
//	res.zderivative.tracers.resize(cells[0].tracers.size(), 0);
	return res;
}

void FreeFlowGenerator3D::operator()(const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells, double /*time*/, boost::container::flat_map<size_t, ComputationalCell3D> &res) const
{
	vector<std::pair<size_t, size_t> > ghosts = GetOuterFacesIndeces(tess);
	size_t N = ghosts.size();
	vector<std::pair<size_t, ComputationalCell3D> > temp(N);
	vector<size_t> indeces(N);
	for (size_t i = 0; i < N; ++i)
	{
		size_t ghost_index = 0, real_index = 0;
		if (ghosts[i].second == 1)
		{
			ghost_index = tess.GetFaceNeighbors(ghosts[i].first).first;
			real_index = tess.GetFaceNeighbors(ghosts[i].first).second;
		}
		else
		{
			ghost_index = tess.GetFaceNeighbors(ghosts[i].first).second;
			real_index = tess.GetFaceNeighbors(ghosts[i].first).first;
		}
		temp[i] = std::pair<size_t, ComputationalCell3D>(ghost_index, cells[real_index]);
		indeces[i] = ghost_index;
	}
	vector<GhostSource> const sources = GetGhostSources(tess, ghosts, indeces);
	res.clear();
	res.reserve(sources.size());
	for (GhostSource const& entry : sources)
		res.insert(res.end(), temp[entry.source]);
}

Slope3D FreeFlowGenerator3D::GetGhostGradient(const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& /*cells*/,
	const vector<Slope3D>& /*gradients*/, size_t /*ghost_index*/, double /*time*/, size_t /*face_index*/) const
{
	Slope3D res;
//	res.xderivative.tracers.resize(cells[0].tracers.size(), 0);
	//res.yderivative.tracers.resize(cells[0].tracers.size(), 0);
	//res.zderivative.tracers.resize(cells[0].tracers.size(), 0);
	return res;
}

ConstantPrimitiveGenerator3D::ConstantPrimitiveGenerator3D(ComputationalCell3D const & cell):cell_(cell) {}

void ConstantPrimitiveGenerator3D::operator()(const Tessellation3D & tess, const vector<ComputationalCell3D>&/*cells*/, 
	double /*time*/, boost::container::flat_map<size_t, 
	ComputationalCell3D>& res) const
{
	vector<std::pair<size_t, size_t> > ghosts = GetOuterFacesIndeces(tess);
	size_t N = ghosts.size();
	vector<std::pair<size_t, ComputationalCell3D> > temp(N);
	vector<size_t> indeces(N);
	for (size_t i = 0; i < N; ++i)
	{
		size_t ghost_index = 0, real_index = 0;
		if (ghosts[i].second == 1)
		{
			ghost_index = tess.GetFaceNeighbors(ghosts[i].first).first;
			real_index = tess.GetFaceNeighbors(ghosts[i].first).second;
		}
		else
		{
			ghost_index = tess.GetFaceNeighbors(ghosts[i].first).second;
			real_index = tess.GetFaceNeighbors(ghosts[i].first).first;
		}
		Vector3D normal = normalize(tess.GetMeshPoint(ghost_index) - tess.GetMeshPoint(real_index));
		temp[i] = std::pair<size_t, ComputationalCell3D>(ghost_index, cell_);
		indeces[i] = ghost_index;
	}
	vector<GhostSource> const sources = GetGhostSources(tess, ghosts, indeces);
	res.clear();
	res.reserve(sources.size());
	for (GhostSource const& entry : sources)
		res.insert(res.end(), temp[entry.source]);
}

Slope3D ConstantPrimitiveGenerator3D::GetGhostGradient(const Tessellation3D & /*tess*/, 
						       const vector<ComputationalCell3D>& /*cells*/, 
						       const vector<Slope3D>& /*gradients*/, size_t /*ghost_index*/, 
	double /*time*/, size_t /*face_index*/) const
{
	Slope3D res;
//	res.xderivative.tracers.resize(cells[0].tracers.size(), 0);
//	res.yderivative.tracers.resize(cells[0].tracers.size(), 0);
//	res.zderivative.tracers.resize(cells[0].tracers.size(), 0);
	return res;
}

SeveralGhostGenerator3D::GhostCriteria3D::~GhostCriteria3D(void){}


SeveralGhostGenerator3D::SeveralGhostGenerator3D(const vector<Ghost3D*>& ghosts, GhostCriteria3D const& ghostchooser) :
	ghosts_(ghosts), ghost_chooser_(ghostchooser){}

void SeveralGhostGenerator3D::operator() (const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells, double time, boost::container::flat_map<size_t, ComputationalCell3D> &res) const 
{
	size_t nghosts = ghosts_.size();
	vector< boost::container::flat_map<size_t, ComputationalCell3D> > ghost_cells(nghosts);
	for (size_t i = 0; i < nghosts; ++i)
		ghosts_[i]->operator()(tess, cells, time, ghost_cells[i]);
	size_t N = ghost_cells[0].size();
	res.clear();
	res.reserve(N);
	for (boost::container::flat_map<size_t, ComputationalCell3D>::const_iterator it = ghost_cells[0].begin(); 
		it != ghost_cells[0].end(); ++it)
	{
		res.insert(std::pair<size_t, ComputationalCell3D>(it->first,
			ghost_cells[ghost_chooser_.GhostChoose(tess, it->first)][it->first]));
	}
}

Slope3D SeveralGhostGenerator3D::GetGhostGradient(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
	const vector<Slope3D>& gradients, size_t ghost_index, double time, size_t face_index) const
{
	return ghosts_[ghost_chooser_.GhostChoose(tess,
		ghost_index)]->GetGhostGradient(tess, cells, gradients, ghost_index, time, face_index);
}

