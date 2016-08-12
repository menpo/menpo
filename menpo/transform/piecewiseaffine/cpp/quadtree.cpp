#include "quadtree.h"


#define TRIANGLE_Y(vertices, index) vertices[index * 2]
#define TRIANGLE_X(vertices, index) vertices[index * 2 + 1]

#define TRIANGLE_I(trilist, index) trilist[index * 3]
#define TRIANGLE_J(trilist, index) trilist[index * 3 + 1]
#define TRIANGLE_K(trilist, index) trilist[index * 3 + 2]


BoundingBox triangle_bounding_box(const unsigned int i, const unsigned int j,
                                  const unsigned int k, const double *vertices) {
    const double y1 = TRIANGLE_Y(vertices, i);
    const double x1 = TRIANGLE_X(vertices, i);
    const double y2 = TRIANGLE_Y(vertices, j);
    const double x2 = TRIANGLE_X(vertices, j);
    const double y3 = TRIANGLE_Y(vertices, k);
    const double x3 = TRIANGLE_X(vertices, k);

    return init_bounding_box(std::min(std::min(y1, y2), y3),
                             std::min(std::min(x1, x2), x3),
                             std::max(std::max(y1, y2), y3),
                             std::max(std::max(x1, x2), x3));
}


bool do_rectangles_overlap(const BoundingBox r1, const BoundingBox r2) {
    return (r1.min_y <= r2.max_y && r1.max_y >= r2.min_y &&
            r1.min_x <= r2.max_x && r1.max_x >= r2.min_x);
}

AlphaBeta point_in_triangle(const unsigned int i, const unsigned int j,
                            const unsigned int k, const double *vertices,
                            const double y, const double x) {
    const double y1 = TRIANGLE_Y(vertices, i);
    const double x1 = TRIANGLE_X(vertices, i);
    const double y2 = TRIANGLE_Y(vertices, j);
    const double x2 = TRIANGLE_X(vertices, j);
    const double y3 = TRIANGLE_Y(vertices, k);
    const double x3 = TRIANGLE_X(vertices, k);

    const double denominator = 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
    const double alpha = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) * denominator;
    const double beta = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) * denominator;

    AlphaBeta ab = {-1, -1};
    if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && alpha + beta <= 1) {
        ab.alpha = alpha;
        ab.beta = beta;
    }
    return ab;
}

void split_bounding_box_into_4(BoundingBox bb, BoundingBox new_boxes[4]) {
    const double half_height = (bb.max_y - bb.min_y) / 2;
    const double half_width = (bb.max_x - bb.min_x) / 2;
    const double center_y = bb.min_y + half_height;
    const double center_x = bb.min_x + half_width;

    const double se_bb_min_y = bb.min_y;
    const double se_bb_min_x = bb.min_x;
    const double ne_bb_min_y = center_y;
    const double ne_bb_min_x = bb.min_x;
    const double nw_bb_min_y = center_y;
    const double nw_bb_min_x = center_x;
    const double sw_bb_min_y = bb.min_y;
    const double sw_bb_min_x = center_x;

    new_boxes[0] = init_bounding_box(se_bb_min_y,
                                     se_bb_min_x,
                                     se_bb_min_y + half_height,
                                     se_bb_min_x + half_width);
    
    new_boxes[1] = init_bounding_box(ne_bb_min_y,
                                     ne_bb_min_x,
                                     ne_bb_min_y + half_height,
                                     ne_bb_min_x + half_width);
    
    new_boxes[2] = init_bounding_box(nw_bb_min_y,
                                     nw_bb_min_x,
                                     nw_bb_min_y + half_height,
                                     nw_bb_min_x + half_width);
    
    new_boxes[3] = init_bounding_box(sw_bb_min_y,
                                     sw_bb_min_x,
                                     sw_bb_min_y + half_height,
                                     sw_bb_min_x + half_width);
}


QuadTree init_quadtree(const BoundingBox bb,
                       const unsigned long depth, const unsigned long max_items,
                       const unsigned long max_depth) {
    QuadTree qtree;
    qtree.bounding_box = bb;
    qtree.depth = depth;
    qtree.max_items = max_items;
    qtree.max_depth = max_depth;
    return qtree;
}

QuadNode init_quadnode(const unsigned long index, const BoundingBox bb) {
    QuadNode qnode;
    qnode.bounding_box = bb;
    qnode.index = index;
    return qnode;
}

BoundingBox init_bounding_box(const double bb_min_y, const double bb_min_x,
                              const double bb_max_y, const double bb_max_x) {
    BoundingBox bb;
    bb.min_y = bb_min_y;
    bb.min_x = bb_min_x;
    bb.max_y = bb_max_y;
    bb.max_x = bb_max_x;
    return bb;
}

bool is_full(const QuadTree &qtree) {
    return qtree.nodes.size() >= qtree.max_items;
}

bool is_max_depth(const QuadTree &qtree) {
    return qtree.depth >= qtree.max_depth;
}

bool insert(QuadTree &qtree, const unsigned long index, const BoundingBox bb) {
    bool insert_successful = true;

    if (qtree.children.empty()) {
        qtree.nodes.push_back(init_quadnode(index, bb));

        if (is_full(qtree)) {
            if (!is_max_depth(qtree)) {
                insert_successful = split(qtree);
            } else {
                insert_successful = false;
            }
        }
    } else {
        insert_successful = insert_into_children(qtree, index, bb);
    }
    return insert_successful;
}

bool insert_into_children(QuadTree &qtree, const unsigned long index, const BoundingBox bb) {
    bool insert_successful = true;

    std::vector<unsigned long> indices;
    for (unsigned int i = 0; i < 4; i++) {
        const QuadTree& c = qtree.children.at(i);
        if (do_rectangles_overlap(c.bounding_box, bb)) {
            indices.push_back(i);
        }
    }

    if (indices.size() == 4) {
        if (!is_full(qtree)) {
            qtree.nodes.push_back(init_quadnode(index, bb));
        } else {
            // Forced to insert this node into every child
            for (QuadTreeIterator it = qtree.children.begin(); it != qtree.children.end(); ++it) {
                insert_successful &= insert(*it, index, bb);
            }
        }
    } else {
        for(std::vector<unsigned long>::iterator it = indices.begin(); it != indices.end(); ++it) {
            insert_successful &= insert(qtree.children.at(*it), index, bb);
        }
    }
    return insert_successful;
}

bool split(QuadTree &qtree) {
    BoundingBox new_boxes[4];
    split_bounding_box_into_4(qtree.bounding_box, new_boxes);

    for(unsigned int i = 0; i < 4; i++) {
        qtree.children.push_back(init_quadtree(new_boxes[i], qtree.depth + 1,
                                               qtree.max_items, qtree.max_depth));
    }

    std::vector<QuadNode> nodes_to_reinsert(qtree.nodes);
    // Clear all nodes from this level as they are now handled by its children
    qtree.nodes.clear();

    bool insert_successful = true;
    for (ConstQuadNodeIterator it = nodes_to_reinsert.begin(); it != nodes_to_reinsert.end(); ++it) {
        const QuadNode& node = *it;
        insert_successful &= insert_into_children(qtree, node.index,
                                                  node.bounding_box);
    }

    return insert_successful;
}

void rect_intersect(const QuadTree &qtree, const BoundingBox bb,
                    std::set<unsigned long>& results) {

    for (ConstQuadTreeIterator it = qtree.children.begin(); it != qtree.children.end(); ++it) {
        const QuadTree& c = *it;
        if (do_rectangles_overlap(c.bounding_box, bb)) {
            rect_intersect(c, bb, results);
        }
    }

    // Search nodes at this level
    if (do_rectangles_overlap(qtree.bounding_box, bb)) {
        for (ConstQuadNodeIterator it = qtree.nodes.begin(); it != qtree.nodes.end(); ++it) {
            const QuadNode& node = *it;
            results.insert(node.index);
        }
    }
}


QuadTree build_quadtree(const unsigned long max_items,
                        const unsigned long max_depth,
                        const double *vertices,
                        const unsigned long n_vertices,
                        const unsigned int *trilist,
                        const unsigned long n_triangles) {

    BoundingBox qtree_bb;
    for (unsigned long i = 0; i < n_vertices; i++) {
        double y = TRIANGLE_Y(vertices, i);
        double x = TRIANGLE_X(vertices, i);
        qtree_bb.min_y = std::min(qtree_bb.min_y, y);
        qtree_bb.min_x = std::min(qtree_bb.min_x, x);
        qtree_bb.max_y = std::max(qtree_bb.max_y, y);
        qtree_bb.max_x = std::max(qtree_bb.max_x, x);
    }
    QuadTree qtree = init_quadtree(qtree_bb, 0, max_items, max_depth);

    for (unsigned long ind = 0; ind < n_triangles; ind++) {
        const unsigned long i = TRIANGLE_I(trilist, ind);
        const unsigned long j = TRIANGLE_J(trilist, ind);
        const unsigned long k = TRIANGLE_K(trilist, ind);
        if (!insert(qtree, ind, triangle_bounding_box(i, j, k, vertices))) {
            dealloc_quadtree(qtree);
            throw std::length_error("Quadtree is full, failed to insert triangle.");
        }
    }

    // Remove empty children to save memory.
    finalize_quadtree(qtree);
    return qtree;
}


void dealloc_quadtree(QuadTree &qtree) {
    qtree.nodes.clear();
    for (QuadTreeIterator it = qtree.children.begin(); it != qtree.children.end(); ++it) {
        dealloc_quadtree(*it);
    }
    qtree.children.clear();
}


void finalize_quadtree(QuadTree &qtree) {
    QuadTreeIterator it = qtree.children.begin();
    while (it != qtree.children.end()) {
        QuadTree& child = *it;
        finalize_quadtree(child);

        if (child.nodes.empty() && child.children.empty()) {
            qtree.children.erase(it);
        } else {
            ++it;
        }
    }
}

