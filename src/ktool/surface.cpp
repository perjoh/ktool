#include "surface.hpp"
#include <algorithm>
#include <array>
#include <glm/geometric.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cassert>

namespace ktool {

    template <typename Vec>
    bool equal_vec(const Vec& a, const Vec& b)
    {
        constexpr auto eps = std::numeric_limits<typename Vec::value_type>::epsilon();
        return glm::all(glm::epsilonEqual(a, b, eps));
    }

    const float Right = 1.0f;
    const float Left = -Right;
    const float Up = 1.0f;
    const float Down = -Up;
    const float Far = 1.0f;
    const float Near = -Far;

    Surface::Surface(const Surface& rhs)
    {
        *this = import_raw(rhs.export_raw());
    }

    Surface::Surface(Surface&& rhs) noexcept
        : vertices_(std::move(rhs.vertices_))
        , edges_(std::move(rhs.edges_))
        , faces_(std::move(rhs.faces_))
        , dirty_indices_(false)
    { 
    }

    //
    Surface Surface::construct_cube()
    {
        Surface cube;

        const auto v0 = cube.add_vertex(glm::vec3(Left,  Up,   Far));
        const auto v1 = cube.add_vertex(glm::vec3(Right, Up,   Far));
        const auto v2 = cube.add_vertex(glm::vec3(Right, Down, Far));
        const auto v3 = cube.add_vertex(glm::vec3(Left,  Down, Far));

        const auto v4 = cube.add_vertex(glm::vec3(Left,  Up,   Near));
        const auto v5 = cube.add_vertex(glm::vec3(Right, Up,   Near));
        const auto v6 = cube.add_vertex(glm::vec3(Right, Down, Near));
        const auto v7 = cube.add_vertex(glm::vec3(Left,  Down, Near));

        const auto top_far    = cube.add_edge(v0, v1);
        const auto right_far  = cube.add_edge(v1, v2);
        const auto bottom_far = cube.add_edge(v2, v3);
        const auto left_far   = cube.add_edge(v3, v0);

        const auto top_near    = cube.add_edge(v4, v5);
        const auto right_near  = cube.add_edge(v5, v6);
        const auto bottom_near = cube.add_edge(v6, v7);
        const auto left_near   = cube.add_edge(v7, v4);

        const auto top_left     = cube.add_edge(v0, v4);
        const auto top_right    = cube.add_edge(v1, v5);
        const auto bottom_right = cube.add_edge(v2, v6);
        const auto bottom_left  = cube.add_edge(v3, v7);

        It_face face = cube.add_face(top_far,     left_far,    bottom_far,   right_far);       // Far
        assert(equal_vec(cube.calc_face_normal(*face), glm::vec3(0, 0, 1)));

        face = cube.add_face(top_near,    right_near,   bottom_near,  left_near);      // Near
        assert(equal_vec(cube.calc_face_normal(*face), glm::vec3(0, 0, -1)));

        face = cube.add_face(top_right,   right_far,    bottom_right, right_near);     // Right
        assert(equal_vec(cube.calc_face_normal(*face), glm::vec3(1, 0, 0)));

        face = cube.add_face(bottom_near, bottom_right, bottom_far,   bottom_left);    // Bottom
        assert(equal_vec(cube.calc_face_normal(*face), glm::vec3(0, -1, 0)));

        face = cube.add_face(top_near,    top_left,     top_far,      top_right);      // Top
        assert(equal_vec(cube.calc_face_normal(*face), glm::vec3(0, 1, 0)));

        face = cube.add_face(top_left,    left_near,    bottom_left,  left_far);       // Left
        assert(equal_vec(cube.calc_face_normal(*face), glm::vec3(-1, 0, 0)));

        return cube;
    }

    //
    Surface Surface::import_raw(const Raw_surface& source)
    {
        const auto& polygons = std::get<0>(source);
        const auto& indices = std::get<1>(source);
        const auto& vertices = std::get<2>(source);

        Surface output;

        std::vector<It_vertex> it_vertices;

        // Add all vertices in the same order as source.
        assert(vertices.size() % 3 == 0);
        for (std::size_t i = 0; i < vertices.size(); i += 3) { 

            auto it = output.add_vertex(glm::vec3(vertices[i], vertices[i+1], vertices[i+2])); 
            it_vertices.push_back(it);
        }

        std::size_t i = 0;
        for (auto polygon : polygons) {

            auto e0 = output.add_edge(it_vertices[indices[i]], it_vertices[indices[i+1]]);
            auto e1 = output.add_edge(it_vertices[indices[i+1]], it_vertices[indices[i+2]]);

            if (polygon == 3) {

                auto e2 = output.add_edge(it_vertices[indices[i+2]], it_vertices[indices[i+0]]); 
                output.add_face(e0, e1, e2);

            } else if (polygon == 4) {

                auto e2 = output.add_edge(it_vertices[indices[i+2]], it_vertices[indices[i+3]]);
                auto e3 = output.add_edge(it_vertices[indices[i+3]], it_vertices[indices[i+0]]); 
                output.add_face(e0, e1, e2, e3);
            }

            i += polygon;
        } 

        return output;
    }

    //
    Surface::Raw_surface Surface::export_raw() const
    {
        assert(!dirty_indices_);

        Raw_surface output;

        auto& polygons = std::get<0>(output);
        auto& indices = std::get<1>(output);
        auto& vertices = std::get<2>(output);

        vertices.reserve(vertices_.size()*3);
        for (const auto& v : vertices_) {
            vertices.push_back(v.position.x);
            vertices.push_back(v.position.y);
            vertices.push_back(v.position.z);
        }

        polygons.reserve(faces_.size());
        for (const auto& face : faces_) {
            polygons.push_back(static_cast<int>(face.edges.size()));

            for (std::size_t i = 0; i < face.edges.size(); ++i) {
                auto edge = face.get_normalized_edge(i);
                indices.push_back(static_cast<int>(edge.vertices[0]->index));
            }
        }

        return output;
    }

    //
    void Surface::scale(const glm::vec3& s)
    {
        for (auto& v : vertices_) {
            v.position *= s;
        }
    }

    //
    void Surface::offset(const glm::vec3& o)
    {
        for (auto& v : vertices_) {
            v.position += o; 
        }
    }

    //
    void Surface::rotate(int axis, float angle_deg) 
    { 
        assert(axis < 3 && !(axis < 0));

        glm::vec3 axis_v;
        axis_v[axis] = 1.0;

        auto mat = glm::rotate(glm::identity<glm::mat4>(), glm::radians(angle_deg), axis_v);

        for (Vertex& vertex : vertices_) {
            vertex.position = mat* glm::vec4{vertex.position, 1.0};
        } 
    }

    // 
    void Surface::skew(int axis, glm::vec3::value_type factor)
    {
        auto min_max = measure(); 
        auto dim = min_max.second - min_max.first;
 
        for (Vertex& vertex : vertices_) { 

            const auto t = (vertex.position[axis] - min_max.first[axis])/dim[axis];

            assert(t <= 1.0);
            assert(t >= 0.0);

            for (int i = 0; i < 3; ++i) {
                if (i != axis) {
                    auto& p = vertex.position[i];
                    p = p*factor*t + p*(1.0f - t);
                } 
            } 
        } 
    }

    //
    std::pair<glm::vec3, glm::vec3> Surface::measure() const
    {
        auto v_min = vertices_.front().position;
        auto v_max = v_min;

        for (const Vertex& vertex : vertices_) {

            v_min.x = std::min(v_min.x, vertex.position.x);
            v_min.y = std::min(v_min.y, vertex.position.y);
            v_min.z = std::min(v_min.z, vertex.position.z);

            v_max.x = std::max(v_max.x, vertex.position.x);
            v_max.y = std::max(v_max.y, vertex.position.y);
            v_max.z = std::max(v_max.z, vertex.position.z);
        }

        return std::make_pair(v_min, v_max);
    }

    //
    bool Surface::operator==(const Surface& rhs) const
    {
        if (   vertices_.size() == rhs.vertices_.size()
            && edges_.size() == rhs.edges_.size()
            && faces_.size() == rhs.faces_.size())
        {
            auto it_b = rhs.vertices_.begin();
            for (auto it_a = vertices_.begin(); it_a != vertices_.end(); ++it_a) {
                constexpr auto eps = std::numeric_limits<glm::vec3::value_type>::epsilon();
                if (!glm::all(glm::epsilonEqual(it_a->position, it_b->position, eps))) {
                    return false;
                }
                ++it_b;
            }

            return extract_soup() == rhs.extract_soup(); 
        }

        return false;
    }
        
    // 
    It_vertex Surface::add_vertex(const glm::vec3& v)
    {
        vertices_.push_back(v);
        auto it = --vertices_.end();
        it->index = vertices_.size()-1;
        return it;
    }

    // Does not create duplicate edges, returns any existing edges.
    It_edge Surface::add_edge(It_vertex vertex_a, It_vertex vertex_b)
    {
        It_edge existing_edge = find_edge(vertex_a, vertex_b);
        if (existing_edge == edges_.end()) {

            edges_.push_back({});

            auto new_edge = --edges_.end();

            new_edge->faces[0] = new_edge->faces[1] = faces_.end();

            new_edge->vertices[0] = vertex_a;
            new_edge->vertices[1] = vertex_b; 

            vertex_a->edges.push_back(new_edge);
            vertex_b->edges.push_back(new_edge);

            return new_edge;
        }

        return existing_edge;
    }

    //
    It_face Surface::add_face(  It_edge edge_a, 
                                It_edge edge_b, 
                                It_edge edge_c, 
                                It_edge edge_d)
    {
        //std::array<It_edge, 4> edges{edge_a, edge_b, edge_c, edge_d};

        faces_.push_back({});
        const auto new_face = --faces_.end();

        new_face->edges.push_back(edge_a);
        new_face->edges.push_back(edge_b);
        new_face->edges.push_back(edge_c);
        if (edge_d != edges_.end()) {
            new_face->edges.push_back(edge_d);
        }

        auto& edges = new_face->edges;

        const unsigned edges_size = edges.size();
        for (unsigned i = 0; i < edges_size; ++i) {

            // Mask any inverted edges.
            if (check_edge(edges[i], edges[(i+1)%edges_size])) {
                edges[i]->faces[SIDE_RIGHT] = new_face; 
            } else {
                // Edge is inverted.
                //new_face->edge_mask |= (1 << i); 
                //new_face->set_edge_inverted(i);
                edges[i]->faces[SIDE_LEFT] = new_face; 
            }
        }

        return new_face;
    }

    //
    void Surface::extrude_face(It_face /*face*/, const glm::vec3& /*direction*/)
    {
    }

    //
    template <typename T>
    std::tuple<T, T> swap_if(T t0, T t1, bool f)
    {
        if (f) {
            return std::make_tuple(t1, t0);
        }

        return std::make_tuple(t0, t1);
    }

    //
    void Surface::sub_divide()
    { 
        using namespace std;

        auto raw = export_raw();

        auto& polygons = get<0>(raw);
        auto& indices = get<1>(raw);
        auto& vertices = get<2>(raw);



        //Surface target;

        /*

        target.vertices_ = vertices_;
        target.vertex_edge_table_.resize(vertex_edge_table_.size());

        // 1. Split edges; add edge points. (Outer edges)
        //std::vector<Edge> split_edges; 
        target.edges_.reserve(edges_.size()*2);
        for (const auto& e : edges_) {
            const glm::vec3 v = (vertices_[e.vertices[0]] + vertices_[e.vertices[1]])*0.5f;
            auto edge_point = target.add_vertex(v);

            target.add_edge(e.vertices[0], edge_point);
            target.add_edge(edge_point, e.vertices[1]); 
        }

        assert(edges_.size()*2 == target.edges_.size());

        // 2. Split faces, add face points.
        target.faces_.reserve(faces_.size()*4);
        for (const auto& face : faces_) {

            assert(edges_connect(face.edges[0], face.edges[1]));
            assert(edges_connect(face.edges[1], face.edges[2]));
            assert(edges_connect(face.edges[2], face.edges[3]));
            assert(edges_connect(face.edges[3], face.edges[0]));

            const auto fp = target.add_vertex(face_point(face));

            {   // Edge points.
                const auto ep0 = target.edges_[edge_index(face.edges[0])*2].vertices[1]; // Second vertex should always be a split edge point.
                const auto ep1 = target.edges_[edge_index(face.edges[1])*2].vertices[1]; // ...
                const auto ep2 = target.edges_[edge_index(face.edges[2])*2].vertices[1]; // ...
                const auto ep3 = target.edges_[edge_index(face.edges[3])*2].vertices[1]; // ...

                // Edges splitting the quad.
                const auto e_top    = target.add_edge(ep0, fp);
                const auto e_right  = target.add_edge(ep1, fp);
                const auto e_bottom = target.add_edge(ep2, fp);
                const auto e_left   = target.add_edge(ep3, fp);

                // Outer edges that was split previously.
                const auto [e_top_a, e_top_b] = swap_if(edge_index(face.edges[0])*2 + 0, 
                                                        edge_index(face.edges[0])*2 + 1,
                                                        inverted_edge(face, 0)); 

                const auto [e_right_a, e_right_b] = swap_if(edge_index(face.edges[1])*2 + 0,
                                                            edge_index(face.edges[1])*2 + 1,
                                                            inverted_edge(face, 1));

                const auto [e_bottom_a, e_bottom_b] = swap_if(edge_index(face.edges[2])*2 + 0,
                                                              edge_index(face.edges[2])*2 + 1,
                                                              inverted_edge(face, 2));

                const auto [e_left_a, e_left_b] = swap_if(edge_index(face.edges[3])*2 + 0,
                                                          edge_index(face.edges[3])*2 + 1,
                                                          inverted_edge(face, 3));

                assert(target.edges_connect(e_top_a, e_top_b));
                assert(target.edges_connect(e_top_b, e_right_a));
                assert(target.edges_connect(e_right_a, e_right_b));
                assert(target.edges_connect(e_right_b, e_bottom_a));
                assert(target.edges_connect(e_bottom_a, e_bottom_b));
                assert(target.edges_connect(e_bottom_b, e_left_a));
                assert(target.edges_connect(e_left_a, e_left_b));
                assert(target.edges_connect(e_left_b, e_top_a));

                // Add faces in a clock-wise order.
                target.add_face(e_top_a, e_top, e_left, e_left_b);
                target.add_face(e_top_b, e_right_a, e_right, e_top);
                target.add_face(e_right, e_right_b, e_bottom_a, e_bottom);
                target.add_face(e_left, e_bottom, e_bottom_b, e_left_a);

                assert(target.edges_connect(e_top_a, e_top));
                assert(target.edges_connect(e_top, e_left));
                assert(target.edges_connect(e_left, e_left_b));
                assert(target.edges_connect(e_top, e_top_a));

                assert(target.edges_connect(e_top_b, e_right_a));
                assert(target.edges_connect(e_right_a, e_right));
                assert(target.edges_connect(e_right, e_top));
                assert(target.edges_connect(e_top, e_top_b));

                assert(target.edges_connect(e_right, e_right_b));
                assert(target.edges_connect(e_right_b, e_bottom_a));
                assert(target.edges_connect(e_bottom_a, e_bottom));
                assert(target.edges_connect(e_bottom, e_right));

                assert(target.edges_connect(e_left, e_bottom));
                assert(target.edges_connect(e_bottom, e_bottom_b));
                assert(target.edges_connect(e_bottom_b, e_left_a));
                assert(target.edges_connect(e_left_a, e_left));
            }

            assert(target.check_vertex_redundancy(fp, 0.05f));
        }

        std::swap(target, *this);

        assert(test_edge_integrity());*/
    }

    //
    void Surface::average()
    {
        std::vector<glm::vec3> new_vertices;
        new_vertices.reserve(vertices_.size());

        for (   It_vertex vertex = vertices_.begin(); 
                vertex != vertices_.end(); 
                ++vertex) {

            float n = 0.0f;
            glm::vec3 face_points_sum(0);
            glm::vec3 edge_points_sum(0);

            It_face first_face = next_face(vertex, faces_.end());
            It_face cur_face = first_face;

            do {
                auto edge = edges_sharing_vertex(cur_face, vertex);

                edge_points_sum += edge_point(std::get<0>(edge));
                //edge_points_sum += edge_point(std::get<1>(edge));

                face_points_sum += face_point(*cur_face);

                n += 1.0f;

                cur_face = next_face(vertex, cur_face);
            } while (cur_face != first_face && cur_face != faces_.end()); 

            // Stolen from Wikipedia, Catmull-Clark subdivision. (https://en.wikipedia.org/wiki/Catmull-Clark_subdivision_surface)
            const auto fp_avg = face_points_sum / n;
            const auto p = vertex->position;
            const auto ep_avg = edge_points_sum / n;
            n *= 2.0f;
            new_vertices.push_back((fp_avg + ep_avg * 2.0f + p * (n - 3.0f)) / n);
            //new_vertices[i_vertex] = glm::vec3(3.0f); // (fp_avg + ep_avg * 2.0f + p * 3.0f) / 6.0f;
            //new_vertices[i_vertex] = (F + edge_points_sum*2.0f + (n - 3.0f)*vertices_[i_vertex])/n;
        }

        std::size_t i = 0;
        for (auto it = vertices_.begin(); it != vertices_.end(); ++it) {
            it->position = new_vertices[i++];
        }
    }

    //
    void Surface::optimize(float tolerance)
    {
        optimize_edge_points(tolerance);
        optimize_face_points(tolerance);
    }

    //
    void Surface::optimize_edge_points(float /*tolerance*/)
    {
        // Remove vertices that have two edges that are perpendicular.
    }

    // 
    void Surface::optimize_face_points(float tolerance)
    {
        // Simple method that should work for most cases. (There's extreme cases that are problematic)
        // 1. Find vertex 'A' with adjacent faces that has normals within tolerance level.
        // 2. Pick any vertex 'B' within these faces that is closest to 'A'.
        // 3. Remove 'A' and create new faces fanning out from 'B'.
        // 4. Repeat 1. until no new candidates are found.

        //to_triangles();

        int num_removed_vertices = 0;
        int num_removed_faces = 0;

        for (auto vertex = vertices_.begin(); vertex != vertices_.end(); ++vertex) { 
            // Setup phase. Populate list with edges fanning out in a clock-wise 
            // order from the vertex.
            auto edge_indices = query::edge_fan(*this, vertex);
            auto edges = query::edge_fan_norm(vertex, edge_indices);
            //auto edges = get_edge_fan_norm(vertex, edge_indices);

            assert(!(edges.size() < 3));

            if (check_vertex_redundancy(edges, tolerance)) { 
                auto shortest_edge = find_shortest_edge(edges);

                auto edge_it = edge_indices[shortest_edge];
                auto vertex_it = edges[shortest_edge].vertices[1];

                //pinch_edge(edge_index(edge_i), vertex_i);

                /*{   // Rotate arrays so first item is that with shortest distance to candidate.
                    using namespace std;
                    auto it = find(edges_i.begin(), edges_i.end(), nearest_vertex);
                    assert(it != edges_i.end());
                    rotate(edges_i.begin(), it, edges_i.end());
                    const size_t offset = std::distance(edges_i.begin(), it);
                    rotate(edges.begin(), edges.begin() + offset, edges.end());
                }*/

                // Traverse edges and create new faces(triangles).
                // (References to faces that will be deleted will be overwritten.)

                // TODO
                /*for (unsigned i = 0; i < edges.size() - 2; ++i) {
                    edges[i].vertices[1] 
                }*/

                // Remove abandoned edges, faces and vertex.
                // TODO
                
                // Done.

                ++num_removed_vertices;
                num_removed_faces += static_cast<int>(edges.size());
            } 
        }

        assert(test_edge_integrity());
    }

    //
    void Surface::to_triangles() {
        
        using namespace std;

        auto raw = export_raw();

        auto& polygons = get<0>(raw);
        auto& indices = get<1>(raw);
        auto& vertices = get<2>(raw);

        Polygons polygons_tri;
        Indices indices_tri;

        size_t i = 0;
        for (size_t poly = 0; poly < polygons.size(); ++poly) 
        {
            if (polygons[poly] == 4) 
            { 
                polygons_tri.push_back(3);
                polygons_tri.push_back(3);

                indices_tri.push_back(indices[i+0]);
                indices_tri.push_back(indices[i+1]);
                indices_tri.push_back(indices[i+2]);

                indices_tri.push_back(indices[i+2]);
                indices_tri.push_back(indices[i+3]);
                indices_tri.push_back(indices[i+0]);

                i += 4;
            } 
            else if (polygons[poly] == 3) 
            { 
                polygons_tri.push_back(3);

                indices_tri.push_back(indices[i++]);
                indices_tri.push_back(indices[i++]);
                indices_tri.push_back(indices[i++]);
            }
        }

        *this = import_raw(make_tuple(move(polygons_tri), move(indices_tri), move(vertices)));

        //for (std::size_t i = 0; i < std:: 

        /*std::size_t counter = faces_.size();

        for (auto& face : faces_) {

            for (std::size_t i = 2; i < face.edges.size(); ++i) { 
                auto edge_a = add_edge(face.get_normalized_edge(0).vertices[0], face.get_normalized_edge(i).vertices[0]);
                auto edge_c = add_edge(face.get_normalized_edge(i).vertices[1], face.get_normalized_edge(0).vertices[0]);
                add_face(edge_a, face.edges[i], edge_c);
            }

            if (--counter == 0) { 
                break;
            }
        }*/



        // Resize vectors to accomodate for new faces and edges.
        /*std::size_t additional_size = 0;
        for (const auto& face : faces_) {
            if (face.is_quad()) {
                ++additional_size;
            }
        }

        faces_.reserve(faces_.size() + additional_size);
        edges_.reserve(edges_.size() + additional_size);
        
        // Iterate by index, iterators will be invalidated.
        for (std::size_t face_i = 0; face_i < faces_.size(); ++face_i) { 
            if (faces_[face_i].is_quad()) {

                auto edge_a = normalized_edge(faces_[face_i].edges[0]);
                auto edge_b = normalized_edge(faces_[face_i].edges[1]);

                auto new_edge = add_edge(edge_b.vertices[1], 
                                         edge_a.vertices[0]);

                // 
                edges_[new_edge].faces[SIDE_RIGHT] = face_i;

                // Remove reference to face.
                {
                    auto& e = edges_[edge_index(faces_[face_i].edges[2])]; 
                    if (e.faces[0] == face_i) {
                        e.faces[0] = Invalid_index;
                    } else {
                        assert(e.faces[1] == face_i);
                        e.faces[1] = Invalid_index;
                    }
                }
                
                // Remove reference to face.
                {
                    auto& e = edges_[edge_index(faces_[face_i].edges[3])];
                    if (e.faces[0] == face_i) {
                        e.faces[0] = Invalid_index;
                    } else {
                        assert(e.faces[1] == face_i);
                        e.faces[1] = Invalid_index;
                    }
                }

                // Note: Might invalidate any references into 'faces_'. Don't keep references or iterators.
                auto new_face = add_face(new_edge, faces_[face_i].edges[2], faces_[face_i].edges[3]);
                assert(inverted_edge(faces_[new_face], 0));
                assert(!faces_[new_face].is_quad());

                // Convert current face to triangle.
                {
                    Face& face = faces_[face_i];
                    face.num_edges = 3;
                    face.edges[2] = new_edge;
                    face.edges[3] = Invalid_index;
                }

                // Make sure our assumptions holds.
                //assert(test_face_edge_integrity(face_i));
                //assert(test_face_edge_integrity(new_face));
            } 
        }
        */
    }



    /*std::vector<Surface::Edge_index> Surface::normalized_edges(const std::vector<Index>& edges_i) const
    {
        std::vector<Edge_index> edges;
        for (const auto& edge_i : edges_i) {
            edges.push_back(std::make_tuple(normalized_edge(edge_i), edge_i));
        }
        return edges;
    }*/


    bool Surface::check_vertex_redundancy(  const std::vector<Edge>& edges, 
                                            float tolerance) const
    {
        /*assert(std::size_t(2) < edges.size());

        // Determine if faces fanning out around vertex has it's normals within the tolerance level.
        const glm::vec3 normal = calc_face_normal(faces_[edges[0].faces[SIDE_RIGHT]]);
        for (Index edge = 1; edge < edges.size(); ++edge) {
            const auto f = edges[edge].faces[SIDE_RIGHT];
            const auto face = faces_[f];
            const auto value = glm::dot(normal, calc_face_normal(face)) + tolerance;
            if (value < 1.0f) {
                return false;
            }
        }*/

        return true;
    }

    bool Surface::check_vertex_redundancy(It_vertex vertex, float tolerance) const
    {
        auto fan = query::edge_fan_norm(*this, vertex);
        return check_vertex_redundancy(fan, tolerance);
    } 

    std::size_t Surface::find_shortest_edge(const std::vector<Edge>& edges) const
    {
        assert(std::size_t(0) < edges.size());

        std::size_t shortest = 0;
        const Edge* edge = &edges[0];
        float min_len = glm::length(edge->vertices[1]->position - edge->vertices[0]->position);
        for (std::size_t i = 1; i < edges.size(); ++i) {

            edge = &edges[i];
            const float len = glm::length(edge->vertices[1]->position - edge->vertices[0]->position);
            if (len < min_len) {
                min_len = len;
                shortest = i; 
            }
        }

        return shortest;
    }


    void Surface::remove_vertex(It_vertex v)
    {
        for (auto& e : v->edges) {
            e->replace_vertex(v, vertices_.end());
        }

        dirty_indices_ = true;

        vertices_.erase(v);
    }


    void Surface::remove_face(It_face f)
    {
        for (auto& e : f->edges) {
            e->replace_face(f, faces_.end());
        }

        faces_.erase(f);
    }


    void Surface::remove_edge(It_edge edge)
    { 
        auto remove_ref = [] (auto& c, It_edge e) {
            auto tmp = std::remove(c.begin(), c.end(), e);
        };

        remove_ref(edge->vertices[0]->edges, edge);
        remove_ref(edge->vertices[1]->edges, edge);
        remove_ref(edge->faces[0]->edges, edge);
        remove_ref(edge->faces[1]->edges, edge);

        edges_.erase(edge);
    }

    unsigned int Surface::vertex_count() const
    {
        return static_cast<unsigned int>(vertices_.size());
    }

    unsigned int Surface::edge_count() const
    {
        return static_cast<unsigned int>(edges_.size());
    }

    unsigned int Surface::face_count() const
    {
        return static_cast<unsigned int>(faces_.size());
    }

    void Surface::update_vertex_indices()
    {
        if (dirty_indices_) {
            for (auto& v : vertices_) {
                v.index = 0;
            }
            //std::size_t i = 0;
            /*for (auto v = vertices_.begin(); v != vertices_.end(); ++v) {
                v->index = i++:
            }*/

            dirty_indices_ = false;
        }
    }

    //
    std::vector<std::size_t> Surface::extract_soup() const
    {
        assert(!dirty_indices_);

        std::vector<std::size_t> soup;

        for (auto& f : faces_) {
            for (auto& e : f.edges) {
                soup.push_back(e->vertices[0]->index);
                soup.push_back(e->vertices[1]->index);
            }
        }

        std::sort(soup.begin(), soup.end());

        return soup;
    }


    //
    It_face Surface::next_face(It_vertex vertex, It_face current_face) const
    {
        It_edge_const edge_it = (current_face == faces_.end())
            ? vertex->edges.front()
            : std::get<0>(edges_sharing_vertex(current_face, vertex));

        // Edge pair returned is in clock-wise order relative to the face.
        // This means in order to traverse to next plane in a clock-wise
        // order, next plane is to the left of the first edge.

        const Edge edge = current_face->get_normalized_edge(edge_it);
        return edge.faces[SIDE_LEFT];
    }

    //
    std::tuple<It_edge_const, It_edge_const> Surface::edges_sharing_vertex(It_face face, It_vertex vertex) const
    {
        It_edge_const first = edges_.end();
        It_edge_const second = edges_.end();

        for (std::size_t i = 0; i < face->edges.size(); ++i) {
            auto edge = face->get_normalized_edge(i);
            if (edge.vertices[0] == vertex) { 
                second = face->edges[i];
            } else if (edge.vertices[1] == vertex) {
                first = face->edges[i];
            }
        }

        return std::make_tuple(first, second);
    }

    // Check direction of edge_a; if second vertex exists in next edge, direction is considered positive.
    bool Surface::check_edge(It_edge edge_a, It_edge edge_b) const
    {
        const auto v = edge_a->vertices[1];
        return v == edge_b->vertices[0] || v == edge_b->vertices[1];
    }

    //
    bool Surface::edges_connect(It_edge edge_a, It_edge edge_b) const
    {
        const auto v0 = edge_a->vertices[0];
        const auto v1 = edge_a->vertices[1];
        const auto v2 = edge_b->vertices[0];
        const auto v3 = edge_b->vertices[1];

        return v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3;
    }

    //
    /*Edge Surface::normalized_edge(Index e) const
    {
        return normalized_edge(edges_, e);
    }*/

    //
    It_edge Surface::find_edge(It_vertex vertex_a, It_vertex vertex_b)
    {
        for (auto edge_it : vertex_a->edges) { 

            if (edge_it->vertices[0] == vertex_b
             || edge_it->vertices[1] == vertex_b) { 

                return edge_it;
            } 
        } 

        return edges_.end();
    }

    //
    /*Edge Surface::normalized_edge(const std::vector<Edge>& edges, Index e)
    {
        assert((e&~Inverted_edge) < edges.size());

        if ((e&Inverted_edge) != 0u) {
            return edges[edge_index(e)].inverted();
        }

        return edges[e];
    }*/

    //
    glm::vec3 Surface::face_point(const Face& face) const
    {
        const auto e0 = face.get_normalized_edge(0);
        const auto e1 = face.get_normalized_edge(1);
        const auto e2 = face.get_normalized_edge(2);

        if (face.edges.size() == 3) { 
            return (e0.vertices[0]->position +
                    e1.vertices[0]->position +
                    e2.vertices[0]->position)
                    / 3.0f;
        }
        else if (face.edges.size() == 4) { 
            const auto e3 = face.get_normalized_edge(3);

            return (e0.vertices[0]->position +
                    e1.vertices[0]->position +
                    e2.vertices[0]->position +
                    e3.vertices[0]->position)
                    * 0.25f;
        }

        return glm::vec3(0);
    }

    //
    glm::vec3 Surface::edge_point(It_edge_const edge) const
    {
        return (edge->vertices[0]->position + edge->vertices[1]->position)*0.5f;
    }

    // TODO Rewrite this. Doesn't work in it's current state.
    void Surface::pinch_edge(It_edge edge_i, It_vertex vertex_keep)
    {
        /*assert(edge_i < edges_.size());
        assert(edges_[edge_i].vertices[0] == vertex_keep 
               || edges_[edge_i].vertices[1] == vertex_keep);

        const auto& edge = edges_[edge_i];
        const Index vertex_remove = edge.other_vertex(vertex_keep);

        // Update edges to reference 'vertex_keep' instead of 'vertex_remove'. (Pinch)
        for (auto&e : vertex_edge_table_[vertex_remove]) {
            edges_[edge_index(e)].replace_vertex(vertex_remove, vertex_keep);
        }

        auto remove_edge_from_face = [this, edge_i] (Index face_i) { 
            if (face_i != Invalid_index) {
                auto& face = faces_[face_i];
                std::remove_if( face.edges.begin(), 
                                face.edges.end(), 
                                [this, edge_i] (Index i) { 
                                    return edge_index(edge_i) == edge_index(i); 
                                });

                --face.num_edges;
            }
        };

        remove_edge_from_face(edge.faces[SIDE_LEFT]);
        remove_edge_from_face(edge.faces[SIDE_RIGHT]);*/


        // OLD ///////

        //
        /*const Index mask = edges_[edge_i].vertices[0] != vertex_keep ? Inverted_edge : 0;
        const Edge edge = normalized_edge(edge_i | mask);

        //
        const Index right_face = edge.faces[SIDE_RIGHT];
        if (right_face != Invalid_index) {

            const Index edge_delete = edge_index(faces_[right_face].next_edge(edge_i));
            const Index edge_keep = edge_index(faces_[right_face].prev_edge(edge_i));

            faces_[edges_[edge_delete].other_face(right_face)].replace_edge(edge_delete, edge_keep);
            edges_[edge_keep].replace_face(right_face, edges_[edge_delete].other_face(right_face));
        }

        //
        const Index left_face = edge.faces[SIDE_LEFT];
        if (left_face != Invalid_index) {

            const Index edge_delete = edge_index(faces_[left_face].prev_edge(edge_i));
            const Index edge_keep = edge_index(faces_[left_face].next_edge(edge_delete));

            faces_[edges_[edge_delete].other_face(left_face)].replace_edge(edge_delete, edge_keep);
            edges_[edge_keep].replace_face(left_face, edges_[edge_delete].other_face(left_face));
        }

        //
        const Index vertex_remove = (mask ? edges_[edge_i].vertices[0] : edges_[edge_i].vertices[1]);
        remove_edge(edge_i);
        remove_vertex(vertex_remove); */
    }

    //
    bool Surface::test_edge_integrity() const
    {
        /*for (const auto& e : edges_) {
            if (       e.vertices[0] == Invalid_index
                    || e.vertices[1] == Invalid_index
                    || e.faces[0] == Invalid_index
                    || e.faces[1] == Invalid_index) {
                return false;
            } 
        } 

        for (Index v = 0; v < vertices_.size(); ++v) {
            for (auto ei : vertex_edge_table_[v]) {
                if (!normalized_edge(ei).has_vertex(v)) {
                    return false;
                }
            }
        }*/

        return true;
    }

    //
    bool Surface::test_face_edge_integrity() const
    {
        // Verify that every normalized edge has 'f' as it's 'RIGHT' face.
        /*for (auto it = faces_.begin(); it != faces_.end(); ++it) {
            if (!test_face_edge_integrity(it)) {
                return false;
            }
        }*/

        return true;
    }

    //
    bool Surface::test_face_edge_integrity(It_face face) const
    {
        /*for (std::size_t i = 0; i < face->edges.size(); ++i) {
            const auto edge = face->get_normalized_edge(i);
            if (edge.faces[SIDE_RIGHT] != face) {
                return false;
            } 
        } */

        return true;
    }

    //
    bool Surface::test_edge_order() const
    { 
        /*for (std::size_t f = 0; f < faces_.size(); ++f) { 

            const auto num_edges = faces_[f].num_edges;

            // Verify that edges are in clock-wise order.
            for (unsigned i = 0; i < faces_[f].num_edges; ++i) {
                const auto edge_0 = normalized_edge(faces_[f].edges[(i+0)%num_edges]);
                const auto edge_1 = normalized_edge(faces_[f].edges[(i+1)%num_edges]); 
                if (edge_0.vertices[1] != edge_1.vertices[0]) {
                    return false;
                }
            }
        }*/

        return true;
    }

    glm::vec3 Surface::calc_face_normal(const Face& face) const
    {
        auto e0 = face.get_normalized_edge(0);
        auto e1 = face.get_normalized_edge(1);
        auto e2 = face.get_normalized_edge(2);

        auto& v0 = e0.vertices[0]->position;
        auto& v1 = e1.vertices[0]->position;
        auto& v2 = e2.vertices[0]->position;

        // Note: Left handed coordinate system.
        return glm::normalize(glm::cross(v1 - v0, v2 - v0));
    }

    //
    void test()
    {
        Surface cube = Surface::construct_cube();
    }

} // namespace ktool
