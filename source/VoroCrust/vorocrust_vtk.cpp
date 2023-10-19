#include "vorocrust_vtk.hpp"

#include "../../source/misc/utils.hpp"
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkPolyhedron.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkPoints.h>
#include <vtkProperty.h>
#include <vtkNew.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSphereSource.h>
#include <vtkAppendPolyData.h>
#include <vtkArrowSource.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkGlyph3D.h>

namespace vorocrust_vtk {
    void write_vtu_PL_Complex(std::filesystem::path const& filename, 
                              PL_Complex const& plc){
        if(filename.extension() != ".vtu"){
            std::cout << "file extension for `filename` in `write_vtu_PL_Complex` must be '.vtu'!!!" << std::endl;
            exit(1);
        }
        auto const& vertices = plc.vertices;
        auto const& faces = plc.faces;

        std::size_t const num_vertices = vertices.size();
        std::size_t const num_faces = faces.size();

        vtkNew<vtkUnstructuredGrid> ugrid;

        vtkNew<vtkPoints> points;
        points->SetNumberOfPoints(num_vertices);

        // set Vertices.
        for (std::size_t p = 0; p < num_vertices; ++p){
            points->SetPoint(vertices[p]->index, vertices[p]->vertex.x, vertices[p]->vertex.y, vertices[p]->vertex.z);
        }
        ugrid->SetPoints(points);

        
        ugrid->Allocate(num_faces);

        // define Faces
        std::vector<vtkIdType> point_array_in_cell;
        for(std::size_t face_index=0; face_index < num_faces; ++face_index){
            vtkNew<vtkIdList> faces_vtk;
            point_array_in_cell.clear();

            auto& face = faces[face_index];
            size_t const Nvertices_in_face = face->vertices.size();

            faces_vtk->InsertNextId(Nvertices_in_face);
            for (std::size_t i = 0; i < Nvertices_in_face; ++i) {
                faces_vtk->InsertNextId(face->vertices[i]->index);
                point_array_in_cell.push_back(face->vertices[i]->index);
            }

            std::sort(point_array_in_cell.begin(), point_array_in_cell.end());
            point_array_in_cell = unique(point_array_in_cell);
            vtkIdType* ptIds = &point_array_in_cell[0];
            ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, faces_vtk->GetPointer(0));        
        }

        std::vector<SurfacePatch> const& patches = plc.patches;

        // set Faces Surface Patch index.
        vtkNew<vtkIntArray> data;
        data->SetName("Patch Index");
        data->SetNumberOfComponents(1);
        data->SetNumberOfValues(num_faces);
        
        for(std::size_t face_index=0; face_index < num_faces; ++face_index){
            data->SetValue(face_index, -2);
        }
        
        for(unsigned int p_index=0; p_index < patches.size(); ++p_index){
            SurfacePatch const& patch = patches[p_index];
            for(VoroCrust::Face const& face : patch){
                data->SetValue(face->index, p_index);
            }
        }

        ugrid->GetCellData()->AddArray(data);

        // write
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        writer->SetCompressionLevel(1);
        writer->SetFileName(filename.c_str());
        writer->SetInputData(ugrid);
        writer->Write();
    }
    
    void write_vtu_faces(std::filesystem::path const& filename, std::vector<std::vector<Vector3D>> const& faces){
        if(filename.extension() != ".vtu"){
            std::cout << "file extension for `filename` in `write_vtu_faces` must be '.vtu'!!!" << std::endl;
            exit(1);
        }

        std::vector<Vector3D> vertices;
        for(auto const& face : faces) 
            vertices.insert(vertices.end(), face.begin(), face.end());
        vertices = unique(vertices);


        std::size_t const num_vertices = vertices.size();
        std::size_t const num_faces = faces.size();

        vtkNew<vtkUnstructuredGrid> ugrid;

        vtkNew<vtkPoints> points;

        points->SetNumberOfPoints(num_vertices);

        // set Vertices.
        for (std::size_t p = 0; p < num_vertices; ++p){
            points->SetPoint(p, vertices[p].x, vertices[p].y, vertices[p].z);
        }
        ugrid->SetPoints(points);

        ugrid->Allocate(num_faces);
        // define Faces
        std::vector<vtkIdType> point_array_in_cell;
        for(std::size_t face_index=0; face_index < num_faces; ++face_index){
            vtkNew<vtkIdList> faces_vtk;
            point_array_in_cell.clear();

            auto& face = faces[face_index];
            size_t const Nvertices_in_face = face.size();

            faces_vtk->InsertNextId(Nvertices_in_face);
            for (std::size_t i = 0; i < Nvertices_in_face; ++i) {
                auto const& it = std::find(vertices.begin(), vertices.end(), face[i]);
                std::size_t const index = std::distance(vertices.begin(), it);
                faces_vtk->InsertNextId(index);
                point_array_in_cell.push_back(index);
            }

            std::sort(point_array_in_cell.begin(), point_array_in_cell.end());
            point_array_in_cell = unique(point_array_in_cell);
            vtkIdType* ptIds = &point_array_in_cell[0];
            ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, faces_vtk->GetPointer(0));        
        }        
        
        // write
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        writer->SetCompressionLevel(1);
        writer->SetFileName(filename.c_str());
        writer->SetInputData(ugrid);
        writer->Write();
    }

    void write_arbitrary_oriented_vectors(std::filesystem::path const& filename, 
                                          std::vector<Vector3D> const& startPoints, 
                                          std::vector<Vector3D> const& vectors, 
                                          std::string const& name, 
                                          double const factor){

        // partly taken from https://stackoverflow.com/questions/59223211/visualize-velocity-field-as-oriented-arrows
        if(filename.extension() != ".vtp"){
            std::cout << "file extension for `filename` in `write_faces_normals` must be '.vtp'!!!" << std::endl;
            exit(1);
        }

        vtkNew<vtkPolyData> polyData;
        vtkNew<vtkPoints> points;

        std::size_t const nPoints = startPoints.size();

        points->SetNumberOfPoints(nPoints);

        for(std::size_t i=0; i<nPoints; ++i)
            points->SetPoint(i, startPoints[i].x, startPoints[i].y,  startPoints[i].z);

        polyData->SetPoints(points);

        vtkNew<vtkDoubleArray> vecs;
        vecs->SetName(name.c_str());
        vecs->SetNumberOfComponents(3);
        vecs->SetNumberOfTuples(nPoints);
        
        for(std::size_t i=0; i< nPoints; ++i)
            vecs->SetTuple3(i, vectors[i].x, vectors[i].y, vectors[i].z);

        polyData->GetPointData()->AddArray(vecs);
        polyData->GetPointData()->SetActiveVectors(name.c_str());

        vtkNew<vtkGlyph3D> glyph;
        glyph->SetInputData(polyData);

        vtkNew<vtkArrowSource> arrow;
        glyph->SetSourceConnection(arrow->GetOutputPort());
        glyph->SetScaleFactor(factor);
        glyph->SetVectorModeToUseVector();
        glyph->Update();


        vtkNew<vtkXMLPolyDataWriter> writer;
        writer->SetCompressionLevel(1);
        writer->SetFileName(filename.c_str());
        writer->SetInputData(glyph->GetOutput());
        writer->Update();
        writer->Write();
    }

    void write_vtu_trees(std::filesystem::path const& filename, 
                         Trees const& trees){
                            
        if(filename.extension() != ".vtu"){
            std::cout << "file extension for `filename` in `write_vtu_trees` must be '.vtu'!!!" << std::endl;
            exit(1);
        }
        std::vector<Vector3D> const& coord_points_vertices = trees.VC_kd_sharp_corners.points;
        std::vector<Vector3D> const& coord_points_edges = trees.VC_kd_sharp_edges.points;
        std::vector<Vector3D> const& coord_points_faces = trees.VC_kd_faces.points;

        std::size_t const num_points_vertices = coord_points_vertices.size();
        std::size_t const num_points_edges = coord_points_edges.size();
        std::size_t const num_points_faces = coord_points_faces.size();

        vtkNew<vtkUnstructuredGrid> ugrid;

        vtkNew<vtkPoints> points;

        points->SetNumberOfPoints(num_points_vertices + num_points_edges + num_points_faces);

        for(std::size_t p = 0; p < num_points_vertices; ++p){
            Vector3D const& point = coord_points_vertices[p];
            points->SetPoint(p, point.x, point.y, point.z);
        }
        
        for(std::size_t p = 0; p < num_points_edges; ++p){
            Vector3D const& point = coord_points_edges[p];
            points->SetPoint(p+num_points_vertices, point.x, point.y, point.z);
        }

        for(std::size_t p = 0; p < num_points_faces; ++p){
            Vector3D const& point = coord_points_faces[p];
            points->SetPoint(p + (num_points_vertices + num_points_edges), point.x, point.y, point.z);
        }

        ugrid->SetPoints(points);
        
        ugrid->Allocate(num_points_vertices + num_points_edges + num_points_faces);
        
        // define Points as Cells
        std::vector<vtkIdType> point_array_in_cell;
        for(std::size_t point_index=0; point_index < (num_points_vertices + num_points_edges + num_points_faces); ++point_index){
            vtkNew<vtkIdList> point_vtk;
            point_array_in_cell.clear();

            point_vtk->InsertNextId(1);
            point_vtk->InsertNextId(point_index);
            point_array_in_cell.push_back(point_index);

            std::sort(point_array_in_cell.begin(), point_array_in_cell.end());

            point_array_in_cell = unique(point_array_in_cell);
            
            vtkIdType* ptIds = &point_array_in_cell[0];
            ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, point_vtk->GetPointer(0));        
        }

        vtkNew<vtkIntArray> data;
        data->SetName("Point On What");
        data->SetNumberOfComponents(1);
        data->SetNumberOfValues(num_points_vertices + num_points_edges + num_points_faces);

        for(std::size_t p = 0; p < num_points_vertices; ++p){
            data->SetValue(p, 0);
        }
        
        for(std::size_t p = 0; p < num_points_edges; ++p){
            data->SetValue(num_points_vertices + p, 1);
        }

        for(std::size_t p = 0; p < num_points_faces; ++p){
            data->SetValue(num_points_edges + num_points_vertices + p, 2);
        }
        
        ugrid->GetCellData()->AddArray(data);

        // write
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        writer->SetCompressionLevel(1);
        writer->SetFileName(filename.c_str());
        writer->SetInputData(ugrid);
        writer->Write();
    }

    void write_nearestNeighbor(std::filesystem::path const& filename, 
                               VoroCrust_KD_Tree const& tree, 
                               Vector3D const& query) {

        if(filename.extension() != ".vtu"){
            std::cout << "file extension for `filename` in `write_nearestNeighbor` must be '.vtu'!!!" << std::endl;
            exit(1);
        }
        std::vector<Vector3D> const& coord_points = tree.points;

        std::size_t num_points = coord_points.size();

        vtkNew<vtkUnstructuredGrid> ugrid;

        vtkNew<vtkPoints> points;    

        points->SetNumberOfPoints(num_points + 1);
        
        for(std::size_t p = 0; p < num_points; ++p){
            Vector3D const& point = coord_points[p];
            points->SetPoint(p, point.x, point.y, point.z);
        }

        points->SetPoint(num_points, query.x, query.y, query.z);

        ugrid->SetPoints(points);
        ugrid->Allocate(num_points + 1);

        // define Points as Cells
        std::vector<vtkIdType> point_array_in_cell;
        for(std::size_t point_index=0; point_index < num_points+1; ++point_index){
            vtkNew<vtkIdList> point_vtk;
            point_array_in_cell.clear();

            point_vtk->InsertNextId(1);
            point_vtk->InsertNextId(point_index);
            point_array_in_cell.push_back(point_index);

            std::sort(point_array_in_cell.begin(), point_array_in_cell.end());

            point_array_in_cell = unique(point_array_in_cell);
            
            vtkIdType* ptIds = &point_array_in_cell[0];
            ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, point_vtk->GetPointer(0));        
        }

        vtkNew<vtkIntArray> data;
        data->SetName("Nearest Neighbor");
        data->SetNumberOfComponents(1);
        data->SetNumberOfValues(num_points+1);

        for(std::size_t p = 0; p < num_points; ++p){
            data->SetValue(p, 0);
        }

        data->SetValue(num_points, 1); // query point

        int index_nearest_neighbor = tree.nearestNeighbor(query);

        data->SetValue(index_nearest_neighbor, 2);

        ugrid->GetCellData()->AddArray(data);
        
        // write
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        writer->SetCompressionLevel(1);
        writer->SetFileName(filename.c_str());
        writer->SetInputData(ugrid);
        writer->Write();
}

void write_kNearestNeighbors(std::filesystem::path const& filename, 
                             VoroCrust_KD_Tree const& tree, 
                             Vector3D const& query, 
                             int k){

    if(filename.extension() != ".vtu"){
        std::cout << "file extension for `filename` in `write_kNearestNeighbors` must be '.vtu'!!!" << std::endl;
        exit(1);
    }
    std::vector<Vector3D> const& coord_points = tree.points;

    std::size_t num_points = coord_points.size();

    vtkNew<vtkUnstructuredGrid> ugrid;

    vtkNew<vtkPoints> points;    

    points->SetNumberOfPoints(num_points + 1);
    
    for(std::size_t p = 0; p < num_points; ++p){
        Vector3D const& point = coord_points[p];
        points->SetPoint(p, point.x, point.y, point.z);
    }

    points->SetPoint(num_points, query.x, query.y, query.z);

    ugrid->SetPoints(points);
    ugrid->Allocate(num_points + 1);

    // define Points as Cells
    std::vector<vtkIdType> point_array_in_cell;
    for(std::size_t point_index=0; point_index < num_points+1; ++point_index){
        vtkNew<vtkIdList> point_vtk;
        point_array_in_cell.clear();

        point_vtk->InsertNextId(1);
        point_vtk->InsertNextId(point_index);
        point_array_in_cell.push_back(point_index);

        std::sort(point_array_in_cell.begin(), point_array_in_cell.end());

        point_array_in_cell = unique(point_array_in_cell);
        
        vtkIdType* ptIds = &point_array_in_cell[0];
        ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, point_vtk->GetPointer(0));        
    }

    vtkNew<vtkIntArray> data;
    data->SetName("Nearest Neighbor");
    data->SetNumberOfComponents(1);
    data->SetNumberOfValues(num_points+1);

    for(std::size_t p = 0; p < num_points; ++p){
        data->SetValue(p, 0);
    }

    data->SetValue(num_points, 1); // query point

    auto const& indices_nearest_neighbors = tree.kNearestNeighbors(query, k);

    for(auto const index_nearest : indices_nearest_neighbors)
        data->SetValue(index_nearest, 2);

    ugrid->GetCellData()->AddArray(data);
    
    // write
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetCompressionLevel(1);
    writer->SetFileName(filename.c_str());
    writer->SetInputData(ugrid);
    writer->Write();
}

void write_radiusSearch(std::filesystem::path const& filename,
                        VoroCrust_KD_Tree const& tree,
                        Vector3D const& query,
                        double const radius){

    if(filename.extension() != ".vtu"){
        std::cout << "file extension for `filename` in `write_kNearestNeighbors` must be '.vtu'!!!" << std::endl;
        exit(1);
    }

    std::vector<Vector3D> const& coord_points = tree.points;

    std::size_t num_points = coord_points.size();

    vtkNew<vtkUnstructuredGrid> ugrid;

    vtkNew<vtkPoints> points;    

    points->SetNumberOfPoints(num_points + 1);
    
    for(std::size_t p = 0; p < num_points; ++p){
        Vector3D const& point = coord_points[p];
        points->SetPoint(p, point.x, point.y, point.z);
    }

    points->SetPoint(num_points, query.x, query.y, query.z);

    ugrid->SetPoints(points);
    ugrid->Allocate(num_points + 1);

    // define Points as Cells
    std::vector<vtkIdType> point_array_in_cell;
    for(std::size_t point_index=0; point_index < num_points+1; ++point_index){
        vtkNew<vtkIdList> point_vtk;
        point_array_in_cell.clear();

        point_vtk->InsertNextId(1);
        point_vtk->InsertNextId(point_index);
        point_array_in_cell.push_back(point_index);

        std::sort(point_array_in_cell.begin(), point_array_in_cell.end());

        point_array_in_cell = unique(point_array_in_cell);
        
        vtkIdType* ptIds = &point_array_in_cell[0];
        ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, point_vtk->GetPointer(0));        
    }

    vtkNew<vtkIntArray> data;
    data->SetName("Nearest Neighbor");
    data->SetNumberOfComponents(1);
    data->SetNumberOfValues(num_points+1);

    for(std::size_t p = 0; p < num_points; ++p){
        data->SetValue(p, 0);
    }

    data->SetValue(num_points, 1); // query point

    auto const& indices_nearest_neighbors = tree.radiusSearch(query, radius);

    for(auto const index_nearest : indices_nearest_neighbors)
        data->SetValue(index_nearest, 2);

    ugrid->GetCellData()->AddArray(data);
    
    // write
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetCompressionLevel(1);
    writer->SetFileName(filename.c_str());
    writer->SetInputData(ugrid);
    writer->Write();
}

void write_nearestNeighborToSegment(std::filesystem::path const& filename,
                                    VoroCrust_KD_Tree const& tree,
                                    std::array<Vector3D, 2> const& segment,
                                    std::size_t const num_points_on_segment){
    
    if(filename.extension() != ".vtu"){
        std::cout << "file extension for `filename` in `write_nearestNeighborToSegment` must be '.vtu'!!!" << std::endl;
        exit(1);
    }

    std::vector<Vector3D> const& coord_points = tree.points;

    std::size_t const num_points_tree = coord_points.size();
    std::size_t const num_points = num_points_tree + num_points_on_segment;

    vtkNew<vtkUnstructuredGrid> ugrid;

    vtkNew<vtkPoints> points;   

    points->SetNumberOfPoints(num_points);

    for(std::size_t p = 0; p < num_points_tree; ++p){
        Vector3D const& point = coord_points[p];
        points->SetPoint(p, point.x, point.y, point.z);
    }

    Vector3D const& x0 = segment[0];
    Vector3D const& v  = segment[1] - segment[0];
    double step = 1.0 / static_cast<double>(num_points_on_segment-1);
    for(std::size_t p = 0; p < num_points_on_segment; ++p){
        Vector3D const& point = x0 + step*p*v;
        points->SetPoint(p+num_points_tree, point.x, point.y, point.z);
    }

    ugrid->SetPoints(points);

    ugrid->Allocate(num_points);

    // define Points as Cells
    std::vector<vtkIdType> point_array_in_cell;
    for(std::size_t point_index=0; point_index < num_points; ++point_index){
        vtkNew<vtkIdList> point_vtk;
        point_array_in_cell.clear();

        point_vtk->InsertNextId(1);
        point_vtk->InsertNextId(point_index);
        point_array_in_cell.push_back(point_index);

        std::sort(point_array_in_cell.begin(), point_array_in_cell.end());

        point_array_in_cell = unique(point_array_in_cell);
        
        vtkIdType* ptIds = &point_array_in_cell[0];
        ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, point_vtk->GetPointer(0));        
    }

    vtkNew<vtkIntArray> data;
    data->SetName("Data");
    data->SetNumberOfComponents(1);
    data->SetNumberOfValues(num_points);

    // set default color for points
    for(std::size_t p = 0; p < num_points_tree; ++p){
        data->SetValue(p, 0);
    }

    // set color of line 
    for(std::size_t p = num_points_tree; p < num_points; ++p){
        data->SetValue(p, 1);
    }

    std::size_t nn_index = tree.nearestNeighborToSegment(segment);
    data->SetValue(nn_index, 2);

    ugrid->GetCellData()->AddArray(data);
        
    // write
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetCompressionLevel(1);
    writer->SetFileName(filename.c_str());
    writer->SetInputData(ugrid);
    writer->Write();
}

void write_ballTree(std::filesystem::path const& filename, 
                    VoroCrust_KD_Tree_Ball const& b_tree){
                        
    if(filename.extension() != ".vtp"){
        std::cout << "file extension for `filename` in `write_ballTree` must be '.vtp'!!!" << std::endl;
        exit(1);
    }

    if(b_tree.empty()) return;

    std::vector<Vector3D> const& centers = b_tree.points;
    std::vector<double> const& ball_radii = b_tree.ball_radii;


    vtkNew<vtkAppendPolyData> appender;
    
    vtkNew<vtkSphereSource> sphere;
    
    int resolution = 7;

    sphere->SetCenter(centers[0].x, centers[0].y, centers[0].z);
    sphere->SetRadius(ball_radii[0]);
    sphere->SetPhiResolution(resolution);
    sphere->SetThetaResolution(resolution);
    
    vtkNew<vtkDoubleArray> data_first;
    data_first->SetName("radius");
    data_first->SetNumberOfComponents(1);
    sphere->Update();

    std::size_t num_of_cells = sphere->GetOutput()->GetNumberOfCells();
    for(std::size_t j = 0; j<num_of_cells; ++j)
        data_first->InsertNextValue(ball_radii[0]);

    sphere->GetOutput()->GetCellData()->SetScalars(data_first);

    appender->SetInputData(sphere->GetOutput());

    for(unsigned int i=1; i < centers.size(); ++i){
        vtkNew<vtkSphereSource> sphere_input;

        sphere_input->SetCenter(centers[i].x, centers[i].y, centers[i].z);
        sphere_input->SetRadius(ball_radii[i]);
        sphere_input->SetPhiResolution(resolution);
        sphere_input->SetThetaResolution(resolution);
        sphere_input->Update();
        
        vtkNew<vtkDoubleArray> data;
        data->SetName("radius");
        data->SetNumberOfComponents(1);
        
        num_of_cells = sphere_input->GetOutput()->GetNumberOfCells();
        for(std::size_t j = 0; j<num_of_cells; ++j)
            data->InsertNextValue(ball_radii[i]);
    
        sphere_input->GetOutput()->GetCellData()->SetScalars(data);

        appender->AddInputData(sphere_input->GetOutput());
    }
    appender->Update();

    vtkNew<vtkXMLPolyDataWriter> writer;
    writer->SetCompressionLevel(1);
    writer->SetFileName(filename.c_str());
    writer->SetInputData(appender->GetOutput());
    writer->Update();
    writer->Write();
}

void write_points(std::filesystem::path const& filename, std::vector<Vector3D> const& point_vectors) {
    if(filename.extension() != ".vtu"){
        std::cout << "file extension for `filename` in `write_points` must be '.vtu'!!!" << std::endl;
        exit(1);
    }

    std::size_t const num_points = point_vectors.size();

    vtkNew<vtkUnstructuredGrid> ugrid;
    vtkNew<vtkPoints> points;

    points->SetNumberOfPoints(num_points);

    for(std::size_t p=0; p<num_points; ++p){
        Vector3D const& point = point_vectors[p];
        points->SetPoint(p, point.x, point.y, point.z); 
    }

    ugrid->SetPoints(points);
    ugrid->Allocate(num_points);
    
    // define Points as Cells
    std::vector<vtkIdType> point_array_in_cell;
    for(std::size_t point_index=0; point_index < num_points; ++point_index){
        vtkNew<vtkIdList> point_vtk;
        point_array_in_cell.clear();

        point_vtk->InsertNextId(1);
        point_vtk->InsertNextId(point_index);
        point_array_in_cell.push_back(point_index);

        std::sort(point_array_in_cell.begin(), point_array_in_cell.end());

        point_array_in_cell = unique(point_array_in_cell);
        
        vtkIdType* ptIds = &point_array_in_cell[0];
        ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, 1, point_vtk->GetPointer(0));        
    }
    
    vtkNew<vtkIntArray> data;
    data->SetName("Separate");
    data->SetNumberOfComponents(1);
    data->SetNumberOfValues(num_points);

    for(std::size_t p=0; p<num_points; ++p){
        data->SetValue(p, point_vectors[p].z > 0 ? 1 : -1);
    }

    ugrid->GetCellData()->AddArray(data);
    
    // write
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetCompressionLevel(1);
    writer->SetFileName(filename.c_str());
    writer->SetInputData(ugrid);
    writer->Write();
}

} // namespace