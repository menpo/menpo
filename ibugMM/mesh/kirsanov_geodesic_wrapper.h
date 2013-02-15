#pragma once

#include<vector>
//#include "exactgeodesic/geodesic_algorithm_dijkstra.h"
//#include "exactgeodesic/geodesic_algorithm_subdivision.h"
#include "exactgeodesic/geodesic_algorithm_exact.h"

class KirsanovGeodesicWrapper {
    public:
        std::vector<double> points;
        std::vector<unsigned> faces;
        geodesic::Mesh mesh;
        KirsanovGeodesicWrapper(double* coords, unsigned n_vertices,
                unsigned* tri_index, unsigned n_triangles);
        ~KirsanovGeodesicWrapper();
        void all_geodesics_from_source_vertices(
                geodesic::GeodesicAlgorithmBase* algorithm,
                unsigned* source_vertices, unsigned n_sources,
                double* phi, unsigned* best_source);
        void all_exact_geodesics_from_source_vertices(
                unsigned* source_vertices, unsigned n_sources, double* phi,
                unsigned* best_source);
//        void all_dijkstra_geodesics_from_source_vertices(
//                unsigned* source_vertices, unsigned n_sources, double* phi,
//                unsigned* best_source);
//        void all_subdivision_geodesics_from_source_vertices(
//                unsigned* source_vertices, unsigned n_sources,
//                double* phi, unsigned* best_source,
//                unsigned subdivision_level);
};

