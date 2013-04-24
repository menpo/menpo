#include "kirsanov_geodesic_wrapper.h"

KirsanovGeodesicWrapper::KirsanovGeodesicWrapper(double* coords, unsigned n_vertices,
                                unsigned* tri_index, unsigned n_triangles) {
    for (unsigned i = 0; i < n_vertices*3; i++) {
        points.push_back(coords[i]);
    }
    for (unsigned i = 0; i < n_triangles*3; i++) {
        faces.push_back(tri_index[i]);
    }
    mesh.initialize_mesh_data(points, faces);
}

KirsanovGeodesicWrapper::~KirsanovGeodesicWrapper() { }

void KirsanovGeodesicWrapper::all_geodesics_from_source_vertices(
        geodesic::GeodesicAlgorithmBase* algorithm,
        unsigned* source_vertices, unsigned n_sources,
        double* phi, unsigned* best_source){
    std::vector<geodesic::SurfacePoint> all_sources;
    for (unsigned i = 0; i < n_sources; i++) {
        geodesic::SurfacePoint source(&mesh.vertices()[source_vertices[i]]);
        all_sources.push_back(source);
    }
    algorithm->propagate(all_sources);
    for (unsigned i = 0; i < mesh.vertices().size(); i++) {
        geodesic::SurfacePoint p(&mesh.vertices()[i]);
        best_source[i] = algorithm->best_source(p, phi[i]);
    }
}

void KirsanovGeodesicWrapper::all_exact_geodesics_from_source_vertices(
        unsigned* source_vertices, unsigned n_sources,
        double* phi, unsigned* best_source){
    geodesic::GeodesicAlgorithmExact algorithm(&mesh);
    all_geodesics_from_source_vertices(&algorithm, source_vertices, n_sources,
            phi, best_source);
}

// currently importing these two headers causes compiler errors
// not critical so come back to fix

//void KirsanovGeodesicWrapper::all_dijkstra_geodesics_from_source_vertices(
//        unsigned* source_vertices, unsigned n_sources,
//        double* phi, unsigned* best_source){
//    geodesic::GeodesicAlgorithmDijkstra algorithm(&mesh);
//    all_geodesics_from_source_vertices(&algorithm, source_vertices, n_sources,
//            phi, best_source);
//}
//
//void KirsanovGeodesicWrapper::all_subdivision_geodesics_from_source_vertices(
//        unsigned* source_vertices, unsigned n_sources,
//        double* phi, unsigned* best_source, unsigned subdivision_level){
//    geodesic::GeodesicAlgorithmSubdivision algorithm(&mesh, subdivision_level);
//    all_geodesics_from_source_vertices(&algorithm, source_vertices, n_sources,
//            phi, best_source);
//}
//
