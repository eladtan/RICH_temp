#include "hdf5_diagnostics.hpp"
#include "../../misc/hdf5_utils.hpp"
#include "../../misc/lazy_list.hpp"

#include "write_vtk.hpp"
#include "write_vtu.hpp"

#ifdef RICH_MPI
#include "../../tessellation/VoronoiMesh.hpp"
#include "mpi/mpi_commands.hpp"
#endif

using namespace H5;

Snapshot::Snapshot(void) :
	mesh_points(),
	cells(),
	time(),
	cycle(),
	tracerstickernames(){}

Snapshot::Snapshot(const Snapshot& source) :
	mesh_points(source.mesh_points),
	cells(source.cells),
	time(source.time),
	cycle(source.cycle),
	tracerstickernames(source.tracerstickernames) 
{}

DiagnosticAppendix::~DiagnosticAppendix(void) {}

namespace 
{

	template<class T> vector<T> read_vector_from_hdf5
		(const Group& file,
			const string& caption,
			const DataType& datatype)
	{
		DataSet dataset = file.openDataSet(caption);
		DataSpace filespace = dataset.getSpace();
		hsize_t dims_out[2];
		filespace.getSimpleExtentDims(dims_out, nullptr);
		const size_t NX = static_cast<size_t>(dims_out[0]);
		vector<T> result(NX);
		dataset.read(&result[0], datatype);
		return result;
	}

	vector<double> read_double_vector_from_hdf5
	(const Group& file, const string& caption)
	{
		return read_vector_from_hdf5<double>
			(file,
				caption,
				PredType::NATIVE_DOUBLE);
	}

	vector<int> read_int_vector_from_hdf5
		(const Group& file,
			const string& caption)
	{
		return read_vector_from_hdf5<int>
			(file,
				caption,
				PredType::NATIVE_INT);
	}
}

namespace {
	class MeshGeneratingPointCoordinate : public LazyList<double>
	{
	public:

		MeshGeneratingPointCoordinate(const Tessellation& tess,
			double Vector2D::* component) :
			tess_(tess), component_(component) {}

		size_t size(void) const
		{
			return static_cast<size_t>(tess_.GetPointNo());
		}

		double operator[](size_t i) const
		{
			return tess_.GetMeshPoint(static_cast<int>(i)).*component_;
		}

	private:
		const Tessellation& tess_;
		double Vector2D::* component_;
	};

	class SingleCellPropertyExtractor
	{
	public:

		virtual double operator()(const ComputationalCell& p) const = 0;

		virtual ~SingleCellPropertyExtractor(void) {}
	};

	class ThermalPropertyExtractor : public SingleCellPropertyExtractor
	{
	public:

		explicit ThermalPropertyExtractor(double ComputationalCell::* var) :
			var_(var) {}

		double operator()(const ComputationalCell& p) const override
		{
			return p.*var_;
		}

	private:
		double ComputationalCell::* var_;
	};

	class CellVelocityComponentExtractor : public SingleCellPropertyExtractor
	{
	public:

		explicit CellVelocityComponentExtractor(double Vector2D::* component) :
			component_(component) {}

		double operator()(const ComputationalCell& p) const override
		{
			return p.velocity.*component_;
		}

	private:
		double Vector2D::* component_;
	};

	class CellsPropertyExtractor : public LazyList<double>
	{
	public:

		CellsPropertyExtractor(const hdsim& sim,
			const SingleCellPropertyExtractor& scpe) :
			sim_(sim), scpe_(scpe) {}

		size_t size(void) const
		{
			return static_cast<size_t>(sim_.getTessellation().GetPointNo());
		}

		double operator[](size_t i) const
		{
			return scpe_(sim_.getAllCells()[i]);
		}

	private:
		const hdsim& sim_;
		const SingleCellPropertyExtractor& scpe_;
	};

	class ConvexHullData
	{
	public:

		vector<double> xvert;
		vector<double> yvert;
		vector<double> nvert;

		explicit ConvexHullData(const Tessellation& tess) :
			xvert(),
			yvert(),
			nvert(static_cast<size_t>(tess.GetPointNo()))
		{
			xvert.reserve(7 * static_cast<size_t>(tess.GetPointNo()));
			yvert.reserve(7 * static_cast<size_t>(tess.GetPointNo()));
			for (int i = 0; i < tess.GetPointNo(); ++i) {
				vector<Vector2D> convhull;
				ConvexHull(convhull, tess, i);
				for (size_t j = 0; j < convhull.size(); ++j) {
					xvert.push_back(convhull[j].x);
					yvert.push_back(convhull[j].y);
				}
				nvert[static_cast<size_t>(i)] = static_cast<int>(convhull.size());
			}
		}
	};

	class StickerSlice : public LazyList<double>
	{
	public:

		StickerSlice(const hdsim& sim,
			size_t index) :
			sim_(sim), index_(index) {}

		size_t size(void) const
		{
			return static_cast<size_t>(sim_.getTessellation().GetPointNo());
		}

		double operator[](size_t i) const
		{
			return static_cast<double>(sim_.getAllCells()[i].stickers[index_]);
		}

	private:
		const hdsim& sim_;
		const size_t index_;
	};

	class TracerSlice : public LazyList<double>
	{
	public:

		TracerSlice(const hdsim& sim,
			size_t index) :
			sim_(sim), index_(index) {}

		size_t size(void) const
		{
			return static_cast<size_t>(sim_.getTessellation().GetPointNo());
		}

		double operator[](size_t i) const
		{
			return sim_.getAllCells()[i].tracers[index_];
		}

	private:
		const hdsim& sim_;
		const size_t index_;
	};
}

void write_snapshot_to_hdf5(hdsim const& sim, string const& fname,
	const vector<DiagnosticAppendix*>& appendices,
	bool const write_vtk, bool const write_vtu)
{
	ConvexHullData chd(sim.getTessellation());
	H5File file(H5std_string(fname), H5F_ACC_TRUNC);
	Group geometry = file.createGroup("/geometry");
	Group gappendices = file.createGroup("/appendices");
	Group hydrodynamic = file.createGroup("/hydrodynamic");
	Group tracers = file.createGroup("/tracers");
	Group stickers = file.createGroup("/stickers");
#ifdef RICH_MPI
	Group mpi = file.createGroup("/mpi");
#endif

	// General
	write_std_vector_to_hdf5
		(file,
			vector<double>(1, sim.getTime()),
			"time");
	write_std_vector_to_hdf5
		(file,
			vector<int>(1, sim.getCycle()),
			"cycle");

	// Geometry  
	write_std_vector_to_hdf5
		(geometry,
			serial_generate
			(MeshGeneratingPointCoordinate
				(sim.getTessellation(), &Vector2D::x)),
			"x_coordinate");
	write_std_vector_to_hdf5
		(geometry,
			serial_generate
			(MeshGeneratingPointCoordinate
				(sim.getTessellation(), &Vector2D::y)),
			"y_coordinate");
	write_std_vector_to_hdf5
		(geometry,
			chd.xvert,
			"x_vertices");
	write_std_vector_to_hdf5
		(geometry,
			chd.yvert,
			"y_vertices");
	write_std_vector_to_hdf5
		(geometry,
			chd.nvert,
			"n_vertices");
	//MPI
#ifdef RICH_MPI
	write_std_vector_to_hdf5
		(mpi,
			serial_generate
			(MeshGeneratingPointCoordinate
				(sim.GetProcTessellation(), &Vector2D::x)),
			"x_coordinate");
	write_std_vector_to_hdf5
		(mpi,
			serial_generate
			(MeshGeneratingPointCoordinate
				(sim.GetProcTessellation(), &Vector2D::y)),
			"y_coordinate");
#endif

	// Hydrodynamic
	write_std_vector_to_hdf5
		(hydrodynamic,
			serial_generate
			(CellsPropertyExtractor
				(sim, ThermalPropertyExtractor(&ComputationalCell::density))),
			"density");
	write_std_vector_to_hdf5
		(hydrodynamic,
			serial_generate
			(CellsPropertyExtractor
				(sim, ThermalPropertyExtractor(&ComputationalCell::pressure))),
			"pressure");
	write_std_vector_to_hdf5
		(hydrodynamic,
			serial_generate
			(CellsPropertyExtractor
				(sim, CellVelocityComponentExtractor(&Vector2D::x))),
			"x_velocity");
	write_std_vector_to_hdf5
		(hydrodynamic,
			serial_generate
			(CellsPropertyExtractor
				(sim, CellVelocityComponentExtractor(&Vector2D::y))),
			"y_velocity");

	const size_t Ntracers = ComputationalCell::tracerNames.size();
	for (size_t i = 0; i < Ntracers; ++i)
	  write_std_vector_to_hdf5(tracers, serial_generate(TracerSlice(sim, i)), ComputationalCell::tracerNames[i]);

	// Stickers
	//	size_t Nstickers = sim.getAllCells().front().stickers.size();
	const size_t Nstickers = ComputationalCell::stickerNames.size();
	for (size_t i = 0; i < Nstickers; ++i)
	  write_std_vector_to_hdf5(stickers, serial_generate(StickerSlice(sim, i)), ComputationalCell::stickerNames[i]);

	// Appendices
	for (size_t i = 0; i < appendices.size(); ++i)
		write_std_vector_to_hdf5
		(gappendices,
			(*(appendices.at(i)))(sim),
			appendices.at(i)->getName());

	//////////////// write vtk/vtu files
	if(write_vtk or write_vtu){
		std::string const file_name_base = "sedov_"+std::to_string(sim.getCycle());
		// make sure this does not create deep copies 
		std::vector<double> const& density = serial_generate(CellsPropertyExtractor(sim, ThermalPropertyExtractor(&ComputationalCell::density)));
		std::vector<double> const& pressure = serial_generate(CellsPropertyExtractor(sim, ThermalPropertyExtractor(&ComputationalCell::pressure)));
		
		std::vector<double> const& velocity_x = serial_generate(CellsPropertyExtractor(sim, CellVelocityComponentExtractor(&Vector2D::x)));
		std::vector<double> const& velocity_y = serial_generate(CellsPropertyExtractor(sim, CellVelocityComponentExtractor(&Vector2D::y)));
		
		std::vector<std::string> const cell_variable_names = {"density", "pressure"};
		// NOTE! this list probably define deep copies - which is a stupid duplication of memory
		std::vector<std::vector<double>> const cell_variables = {density, pressure};
		
		std::vector<std::string> const cell_vectors_names = {"velocity"};
		// NOTE! this list probably define deep copies - which is a stupid duplication of memory
		std::vector<std::vector<double>> const cell_vectors_x = {velocity_x};
		std::vector<std::vector<double>> const cell_vectors_y = {velocity_y};
		
		std::vector<std::size_t> cells_num_vertices(chd.nvert.size());
		for (size_t cell = 0; cell < chd.nvert.size(); ++cell){
			std::size_t const num_vertices = static_cast<std::size_t>(chd.nvert[cell]);
			assert(num_vertices == chd.nvert[cell]); //make sure it is an integer and no overflow from double->integer
			cells_num_vertices[cell] = num_vertices;
		}

		if(write_vtk){
			write_vtk::write_vtk(
				file_name_base,
				chd.xvert,
				chd.yvert,
				cells_num_vertices,
				cell_variable_names,
				cell_variables,
				sim.getTime(), 
				sim.getCycle()
			);
		}
		
		if(write_vtu){
			write_vtu::write_vtu(
				file_name_base,
				chd.xvert,
				chd.yvert,
				cells_num_vertices,
				cell_variable_names,
				cell_variables,

				cell_vectors_names,
				cell_vectors_x,
				cell_vectors_y,

				sim.getTime(), 
				sim.getCycle()
			);
		}
	}
}

Snapshot read_hdf5_snapshot
(const string& fname
#ifdef RICH_MPI
, bool mpioverride
#endif // RICH_MPI
)
{
	Snapshot res;
	H5File file(fname, H5F_ACC_RDONLY);
	Group g_geometry = file.openGroup("geometry");
	Group g_hydrodynamic = file.openGroup("hydrodynamic");
	Group g_tracers = file.openGroup("tracers");
	Group g_stickers = file.openGroup("stickers");
#ifdef RICH_MPI
	Group mpi;
	if (!mpioverride)
		mpi = file.openGroup("/mpi");
#else
	//	if (mpioverride)
	//		mpioverride = true;
#endif


	// Mesh points
	{
		const vector<double> x =
			read_double_vector_from_hdf5(g_geometry, "x_coordinate");
		const vector<double> y =
			read_double_vector_from_hdf5(g_geometry, "y_coordinate");
		res.mesh_points.resize(x.size());
		for (size_t i = 0; i < x.size(); ++i)
			res.mesh_points.at(i) = Vector2D(x.at(i), y.at(i));
	}

#ifdef RICH_MPI
	// MPI
	{
		if (!mpioverride)
		{
			const vector<double> x =
				read_double_vector_from_hdf5(mpi, "x_coordinate");
			const vector<double> y =
				read_double_vector_from_hdf5(mpi, "y_coordinate");
			res.proc_points.resize(x.size());
			for (size_t i = 0; i < x.size(); ++i)
				res.proc_points.at(i) = Vector2D(x.at(i), y.at(i));
		}
	}
#endif

	// Hydrodynamic
	{
		const vector<double> density =
			read_double_vector_from_hdf5(g_hydrodynamic, "density");
		const vector<double> pressure =
			read_double_vector_from_hdf5(g_hydrodynamic, "pressure");
		const vector<double> x_velocity =
			read_double_vector_from_hdf5(g_hydrodynamic, "x_velocity");
		const vector<double> y_velocity =
			read_double_vector_from_hdf5(g_hydrodynamic, "y_velocity");

		vector<vector<double> > tracers(g_tracers.getNumObjs());
		vector<string> tracernames(tracers.size());
		for (hsize_t n = 0; n < g_tracers.getNumObjs(); ++n) 
		{
			const H5std_string name = g_tracers.getObjnameByIdx(n);
			tracernames[n] = name;
			tracers[n]= read_double_vector_from_hdf5(g_tracers, name);
		}

		vector<vector<int> > stickers(g_stickers.getNumObjs());
		vector<string> stickernames(stickers.size());
		for (hsize_t n = 0; n < g_stickers.getNumObjs(); ++n) 
		{
			const H5std_string name = g_stickers.getObjnameByIdx(n);
			stickernames[n] = name;
			stickers[n] =read_int_vector_from_hdf5(g_stickers, name);
		}
		res.tracerstickernames.first = tracernames;
		res.tracerstickernames.second = stickernames;
		res.cells.resize(density.size());
		for (size_t i = 0; i < res.cells.size(); ++i) 
		{
			res.cells.at(i).density = density.at(i);
			res.cells.at(i).pressure = pressure.at(i);
			res.cells.at(i).velocity.x = x_velocity.at(i);
			res.cells.at(i).velocity.y = y_velocity.at(i);
	//		res.cells.at(i).tracers.resize(tracernames.size());
			for (size_t j = 0; j < tracernames.size(); ++j)
				res.cells.at(i).tracers.at(j) = tracers.at(j).at(i);
		//	res.cells.at(i).stickers.resize(stickernames.size());
			for (size_t j = 0; j < stickernames.size(); ++j)
				res.cells.at(i).stickers.at(j) = stickers.at(j).at(i)==1;
		}
	}

	// Misc
	{
		const vector<double> time =
			read_double_vector_from_hdf5(file, "time");
		res.time = time.at(0);
		const vector<int> cycle =
			read_int_vector_from_hdf5(file, "cycle");
		res.cycle = cycle.at(0);
	}

	return res;
}

void WriteDelaunay(Delaunay const& tri, string const& filename)
{
	vector<Vector2D> const& cor = tri.getCor();
	vector<double> x_cor, y_cor;
	vector<int> facets;
	size_t nfacets = tri.get_num_facet();

	H5File file(H5std_string(filename), H5F_ACC_TRUNC);

	for (size_t i = 0; i < cor.size(); ++i)
	{
		x_cor.push_back(cor[i].x);
		y_cor.push_back(cor[i].y);
	}

	for (size_t i = 0; i < nfacets; ++i)
	{
	  facets.push_back(static_cast<int>(tri.get_facet(i).vertices.first));
	  facets.push_back(static_cast<int>(tri.get_facet(i).vertices.second));
	  facets.push_back(static_cast<int>(tri.get_facet(i).vertices.third));
	}

	write_std_vector_to_hdf5(file, x_cor, "x_coordinate");
	write_std_vector_to_hdf5(file, y_cor, "y_coordinate");
	write_std_vector_to_hdf5(file, vector<int>(1, static_cast<int>(tri.GetOriginalLength())), "point number");
	write_std_vector_to_hdf5(file, facets, "triangles");
}

#ifdef RICH_MPI

Snapshot ReDistributeData(string const& filename, Tessellation const& proctess, size_t snapshot_number)
{
	int ws, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double read_num =
		static_cast<double>(snapshot_number)*1.0 /
		static_cast<double>(ws);
	// Read the data
	int start = static_cast<int>
		(floor
			(static_cast<double>(rank)*read_num + 0.1));
	int stop = static_cast<int>
		(floor((1 + rank)*read_num - 0.9));
	Snapshot res, snap;
	for (int i = start; i < stop; ++i)
	{
		Snapshot temp = read_hdf5_snapshot(filename + int2str(i) + ".h5");
		snap.cells.insert(snap.cells.end(), temp.cells.begin(), temp.cells.end());
		snap.mesh_points.insert(snap.mesh_points.end(), temp.mesh_points.begin(), temp.mesh_points.end());
		if (i == start)
		{
			snap.time = temp.time;
			snap.cycle = temp.cycle;
			snap.tracerstickernames = temp.tracerstickernames;
		}
	}
	vector<vector<Vector2D> > chull(static_cast<size_t>(ws));
	vector<vector<size_t> > indeces(chull.size());
	for (size_t i = 0; i < chull.size(); ++i)
		ConvexHull(chull[i], proctess, static_cast<int>(i));
	for (size_t i = 0; i < snap.mesh_points.size(); ++i)
	{
		bool added = false;
		for (size_t j = 0; j < chull.size(); ++j)
		{
			if (PointInCell(chull[j], snap.mesh_points[i]))
			{
				indeces[j].push_back(i);
				added = true;
				break;
			}
		}
		if (!added)
			throw UniversalError("Didn't find point in ReDistributeData");
	}
	// Send/Recv data
	std::vector<std::vector<ComputationalCell>> cell_recv = MPI_exchange_data_indexed(std::vector<rank_t>(ws), snap.cells, indeces);
	std::vector<std::vector<Vector2D>> mesh_recv = MPI_exchange_data_indexed(std::vector<rank_t>(ws), snap.mesh_points, indeces);

	for(size_t i = 0; i < chull.size(); ++i)
	{
		if(i == static_cast<size_t>(rank))
		{
			continue;
		}
		res.cells.insert(res.cells.end(), cell_recv[i].begin(), cell_recv[i].end());
		res.mesh_points.insert(res.mesh_points.end(), mesh_recv[i].begin(), mesh_recv[i].end());
	}

	res.time = snap.time;
	res.cycle = snap.cycle;
	res.tracerstickernames = snap.tracerstickernames;
	return res;
}

Snapshot ReDistributeData2(string const& filename, Tessellation const& proctess, size_t snapshot_number, bool mpioverride)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Read the data
	Snapshot snap;
	for (int i = 0; i < static_cast<int>(snapshot_number); ++i)
	{
		Snapshot temp = read_hdf5_snapshot(filename + int2str(i) + ".h5", mpioverride);
		snap.cells.insert(snap.cells.end(), temp.cells.begin(), temp.cells.end());
		snap.mesh_points.insert(snap.mesh_points.end(), temp.mesh_points.begin(), temp.mesh_points.end());
		if (i == 0)
		{
			snap.time = temp.time;
			snap.cycle = temp.cycle;
			snap.tracerstickernames = temp.tracerstickernames;
		}
	}
	vector<Vector2D> chull;
	ConvexHull(chull, proctess, rank);
	vector<size_t> indeces;
	for (size_t i = 0; i < snap.mesh_points.size(); ++i)
		if (PointInCell(chull, snap.mesh_points[i]))
			indeces.push_back(i);
	snap.mesh_points = VectorValues(snap.mesh_points, indeces);
	snap.cells = VectorValues(snap.cells, indeces);
	return snap;
}
#endif

void WriteTess(Tessellation const& tess, string const& fname)
{
	ConvexHullData chd(tess);
	H5File file(H5std_string(fname), H5F_ACC_TRUNC);
	Group geometry = file.createGroup("/geometry");
	write_std_vector_to_hdf5(geometry, serial_generate(MeshGeneratingPointCoordinate(tess, &Vector2D::x)),
		"x_coordinate");
	write_std_vector_to_hdf5(geometry, serial_generate(MeshGeneratingPointCoordinate(tess, &Vector2D::y)),
		"y_coordinate");
	write_std_vector_to_hdf5(geometry, chd.xvert, "x_vertices");
	write_std_vector_to_hdf5(geometry, chd.yvert, "y_vertices");
	write_std_vector_to_hdf5(geometry, chd.nvert, "n_vertices");
}
