#pragma once
#ifndef QUADTREE_H
#define QUADTREE_H

#include <set>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>


struct AlphaBeta {
    double alpha;
    double beta;
};

struct Point {
    double x;
    double y;
};

inline Point operator+(Point lhs, const Point& rhs) {
    // passing lhs by value helps optimize chained a+b+c
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs; // return the result by value (uses move constructor)
}

struct BoundingBox {
    Point min;
    Point max;
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

bool insert(QuadTree &qtree, const unsigned long index, const BoundingBox bb);

bool insert_into_children(QuadTree &qtree, const unsigned long index,
                          const BoundingBox bb);

bool split(QuadTree &qtree);

void point_intersect(const QuadTree &qtree, const Point p,
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


inline QuadTree init_quadtree(const BoundingBox bb,
                              const unsigned long depth,
                              const unsigned long max_items,
                              const unsigned long max_depth) {
    QuadTree qtree;
    qtree.bounding_box = bb;
    qtree.depth = depth;
    qtree.max_items = max_items;
    qtree.max_depth = max_depth;
    return qtree;
}

inline QuadNode init_quadnode(const unsigned long index, const BoundingBox bb) {
    QuadNode qnode;
    qnode.bounding_box = bb;
    qnode.index = index;
    return qnode;
}

inline Point init_point(const double x, const double y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

inline BoundingBox init_bounding_box(const Point min, const Point max) {
    BoundingBox bb;
    bb.min = min;
    bb.max = max;
    return bb;
}

inline bool is_full(const QuadTree &qtree) {
    return qtree.nodes.size() >= qtree.max_items;
}

inline bool is_max_depth(const QuadTree &qtree) {
    return qtree.depth >= qtree.max_depth;
}

inline bool approx_equal(const double a, const double b, const double abs = 1e-15, const double rel = 1e-10) {
    // Check if the numbers are really close -- needed when comparing numbers near zero.
    double diff = fabs(a - b);
    if (diff <= abs) return true;

    // Otherwise fall back to Knuth's algorithm
    return diff <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * rel);
}

#endif
