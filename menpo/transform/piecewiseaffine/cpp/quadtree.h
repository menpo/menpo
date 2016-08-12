#pragma once
#ifndef QUADTREE_H
#define QUADTREE_H

#include <set>
#include <algorithm>
#include <stdexcept>
#include <vector>


struct AlphaBeta {
    double alpha;
    double beta;
};


struct BoundingBox {
    double min_y;
    double min_x;
    double max_y;
    double max_x;
};


struct QuadNode {
    unsigned long index;
    BoundingBox bounding_box;
};


struct QuadTree {
    BoundingBox bounding_box;
    unsigned long depth;
    unsigned long max_items;
    unsigned long max_depth;
    std::vector<QuadNode> nodes;
    std::vector<QuadTree> children;
};

typedef std::vector<QuadNode>::const_iterator ConstQuadNodeIterator;
typedef std::vector<QuadTree>::const_iterator ConstQuadTreeIterator;
typedef std::vector<QuadTree>::iterator QuadTreeIterator;


bool do_rectangles_overlap(const BoundingBox r1, const BoundingBox r2);

AlphaBeta point_in_triangle(const unsigned int i, const unsigned int j,
                            const unsigned int k, const double *vertices,
                            const double y, const double x);

void split_bounding_box_into_4(const BoundingBox bb,
                               const BoundingBox new_boxes[4]);


QuadTree init_quadtree(const BoundingBox bb,
                       const unsigned long depth, const unsigned long max_items,
                       const unsigned long max_depth);

QuadNode init_quadnode(const unsigned long index, const BoundingBox bb);

BoundingBox init_bounding_box(const double bb_min_y, const double bb_min_x,
                              const double bb_max_y, const double bb_max_x);

bool is_full(const QuadTree &qtree);

bool is_max_depth(const QuadTree &qtree);

bool insert(QuadTree &qtree, const unsigned long index, const BoundingBox bb);

bool insert_into_children(QuadTree &qtree, const unsigned long index,
                          const BoundingBox bb);

bool split(QuadTree &qtree);

void rect_intersect(const QuadTree &qtree, const BoundingBox bb,
                    std::set<unsigned long>& results);

BoundingBox triangle_bounding_box(const unsigned int i, const unsigned int j,
                                  const unsigned int k, const double *vertices);

QuadTree build_quadtree(const unsigned long max_items,
                        const unsigned long max_depth,
                        const double *vertices,
                        const unsigned long n_vertices,
                        const unsigned int *trilist,
                        const unsigned long n_triangles);

void dealloc_quadtree(QuadTree &qtree);

void finalize_quadtree(QuadTree &qtree);

#endif
