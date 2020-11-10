#pragma once
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <tuple>
#include <algorithm>
#include <array>
#include <list>

namespace ktool {

    struct Plane
    {
        Plane(const glm::vec3& normal_, glm::vec3::value_type d_)
            : normal(normal_)
            , d(d_)
        { 
        }

        Plane(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
            : normal(glm::normalize(glm::cross(b - a, c - a)))
            , d(glm::dot(normal, a))
        { 
        }

        bool intersect_segment(const glm::vec3& a, const glm::vec3& b, glm::vec3& q) const
        {
            // See: Real-Time Collision Detection, page 176.

            const auto ab = b - a;
            const auto t = (d - glm::dot(normal, a)) / glm::dot(normal, ab);

            if (!(t < 0.0) && t <= 1.0) {
                q = a + ab * t;
                return true;
            }

            return false; 
        }

        glm::vec3 normal;
        glm::vec3::value_type d; 
    };


    //
    enum { SIDE_LEFT, SIDE_RIGHT }; 
    
    struct Edge;
    using It_edge = std::list<Edge>::iterator;
    using It_edge_const = std::list<Edge>::const_iterator;

    struct Vertex;
    using It_vertex = std::list<Vertex>::iterator;

    struct Face;
    using It_face = std::list<Face>::iterator;

    struct Vertex
    {
        Vertex() = default;
        Vertex(const glm::vec3& p)
            : position(p)
        { }

        glm::vec3 position;
        std::vector<It_edge> edges;
        std::size_t index{0}; // Used for export.
    };

    // 
    struct Edge 
    { 
        std::array<It_vertex, 2> vertices;
        std::array<It_face, 2> faces;

        //
        bool has_vertex(It_vertex v) const {
            return vertices[0] == v || vertices[1] == v;
        }

        //
        It_vertex other_vertex(It_vertex v) const {
            if (vertices[0] == v) {
                return vertices[1];
            } else if (vertices[1] == v) {
                return vertices[0];
            }

            return It_vertex();
        }

        // Check if edges connects to another edge. (Shares vertex)
        bool connects(const Edge& e) const {
            return has_vertex(e.vertices[0]) || has_vertex(e.vertices[1]);
        }

        // Return an inverted instance of itself. Used for "normalized" edges.
        Edge inverted() const {
            Edge e(*this);
            std::swap(e.vertices[0], e.vertices[1]);
            std::swap(e.faces[0], e.faces[1]);
            return e;
        }

        //
        void replace_vertex(It_vertex from, It_vertex to)
        {
            if (vertices[0] == from) {
                vertices[0] = to;
            } else if (vertices[1] == from) {
                vertices[1] = to;
            }
        }

        //
        void replace_face(It_face from, It_face to)
        {
            if (faces[0] == from) {
                faces[0] = to;
            } else if (faces[1] == from) {
                faces[1] = to;
            }
        }

        //
        It_face other_face(It_face face) const
        {
            if (faces[0] == face) {
                return faces[1];
            } else if (faces[1] == face) {
                return faces[0];
            }

            return It_face();
        }

        // Return this edge but make sure it originates from provided vertex.
        Edge aligned_edge(It_vertex vertex) const
        {
            assert(vertices[0] == vertex || vertices[1] == vertex);

            if (vertices[0] != vertex) {
                return inverted();
            }

            return *this;
        }
    };


    // Either a triangle or a quad.
    struct Face 
    {
        // Stored in clockwise order.
        std::vector<It_edge> edges; 

        //
        bool is_quad() const { return edges.size() == 4; }

        //
        Edge get_normalized_edge(std::size_t i) const
        { 
            auto second_vertex = edges[i]->vertices[1];
            auto next_edge = edges[(i+1)%edges.size()];
            return next_edge->has_vertex(second_vertex) ? *edges[i]
                                                        : edges[i]->inverted();
        }

        Edge get_normalized_edge(It_edge_const edge) const
        {
            auto it = std::find(edges.begin(), edges.end(), edge);
            assert(it != edges.end());
            return get_normalized_edge(it - edges.begin());
        }

        // From a given edge index, return index to edge next in order. (Clock-wise)
        // If provided edge index is invalid, return first edge index.
        It_edge next_edge(It_edge edge, std::size_t offset = 1) const
        {
            std::size_t i = std::find(edges.begin(), edges.end(), edge) - edges.begin();
            return edges[(i + offset)%edges.size()];

        }

        // From a given edge index, return index to edge previous in order. (Counter clock-wise)
        // If provided edge index is invalid, return first edge index.
        It_edge prev_edge(It_edge edge) const
        {
            return next_edge(edge, edges.size() - 1);
        }

        //
        void replace_edge(It_edge from, It_edge to)
        {
            auto it = std::find(edges.begin(), edges.end(), from);
            if (it != edges.end()) {
                *it = to;
            }
        }
    };


    struct query;

    // 
    class Surface
    {
        friend struct query;

    public :
        Surface() = default;
        Surface(const Surface&);
        Surface(Surface&&) noexcept;

        static Surface construct_cube();

        // Raw polygon data:
        // Polygons - Array of integers that determine if the polygon is a triangle or a quad. Either 3 or 4.
        // Indices - Indexes to vertices in the polygon. Grouped either by 3 or 4 determined by the value in 'polygons'.
        // Vertices - Three values for each vertex. (xyz)
        using Polygons = std::vector<int>;
        using Indices = std::vector<int>;
        using Vertices = std::vector<float>;
        using Raw_surface = std::tuple<Polygons, Indices, Vertices>;

        // Import shape from raw polygon data.
        [[nodiscard]]
        static Surface import_raw(const Raw_surface&);

        // 
        Raw_surface export_raw() const;

    public :
        void scale(const glm::vec3&);
        void offset(const glm::vec3&);
        void rotate(int axis, float angle_deg);
        void skew(int axis, float factor);

        std::pair<glm::vec3, glm::vec3> measure() const;

    public :
        bool operator==(const Surface&) const;

        Surface& operator=(Surface&&) = default;

    private :
        //using It_vertex = std::list<Vertex>::iterator;
        It_vertex add_vertex(const glm::vec3&);

        //using It_edge = std::list<Edge>::iterator;
        It_edge add_edge(It_vertex a, It_vertex b);

        //using It_face = std::list<Face>::iterator;
        It_face add_face(It_edge a, It_edge b, It_edge c)
        {
            return add_face(a, b, c, edges_.end());
        }

        It_face add_face(It_edge a, It_edge b, It_edge c, It_edge d);

    public :
        //
        void extrude_face(It_face face, const glm::vec3& direction);

    public :
        //
        void sub_divide();
        void average();

    public :
        void optimize(float tolerance = 0.05f);

        void optimize_edge_points(float tolerance = 0.05f);

        // Identifies "face points", points on faces that can be removed 
        // without affecting the overall shape of the surface, and removes them.
        void optimize_face_points(float tolerance = 0.05f);

        // Convert quads to triangles.
        void to_triangles();

    private :
        // Convert array of edge indices to array of normalized edges.
        //using Edge_index = std::tuple<Edge, Index>;
        //std::vector<Edge_index> normalized_edges(const std::vector<Index>& edges) const;

        // Check if vertex is redundant by checking surrounding faces normals similarity.
        // @edges Normalized edges that share a vertex.
        // @tolerance Tolerance level used to determine normal similarity.
        bool check_vertex_redundancy(const std::vector<Edge>& edges, float tolerance) const;
        bool check_vertex_redundancy(It_vertex vertex, float tolerance) const;

        // From a set of edges, return the one that has the shortest distance between it's vertices.
        std::size_t find_shortest_edge(const std::vector<Edge>& edges) const;

    private :
        void remove_vertex(It_vertex v);
        void remove_face(It_face f);
        void remove_edge(It_edge e);

    public :
        unsigned int vertex_count() const;
        unsigned int edge_count() const;
        unsigned int face_count() const;

    public :
        // 
        void update_vertex_indices();

        //
        template <typename Consumer>
        void export_vertices(Consumer& c)
        {
            for (const auto& v : vertices_) {
                const auto& pos = v.position;
                c(pos.x, pos.y, pos.z);
            } 
        }

        //
        /*template <typename Consumer>
        void export_triangles(Consumer& c)
        {
            for (const auto& q : faces_) {

                const auto e0 = normalized_edge(q.edges[0]);
                const auto e1 = normalized_edge(q.edges[1]);
                const auto e2 = normalized_edge(q.edges[2]);
                const auto e3 = normalized_edge(q.edges[3]);

                c(e0[0], e1[0], e2[0]);
                c(e2[0], e3[0], e0[0]);
            }
        }*/

    private :
        //struct Vertex { float x, y, z; };
        std::list<Vertex> vertices_; 
        std::list<Edge> edges_;
        std::list<Face> faces_;

        bool dirty_indices_{false};

        // 
        //std::map<It_vertex, std::vector<It_edge>> vertex_edge_table_;

        // For testing purposes.
        // Returns a soup of indices that can be used for comparison with likewise objects for other surfaces.
        // In short, used to compare two shapes.
        std::vector<std::size_t> extract_soup() const;

    public :
        // Returns the next, clock-wise face sharing the supplied vertex.
        // If 'current_face' equals Invalid_vertex, an edge is picked from 'vertex_edge_table_'
        // and face traversal begins from there.
        It_face next_face(It_vertex vertex, It_face current_face) const;

        // Return two edges in face that shares the same vertex. 
        // Note: Edges are in clock-wise order.
        // TODO: Move to Face.
        std::tuple<It_edge_const, It_edge_const> edges_sharing_vertex(It_face face, It_vertex vertex) const;

        // Returns true if edge is not inverted.
        bool check_edge(It_edge edge_a, It_edge edge_b) const;

        // Determines if supplied edges connect. Used for integrity checks.
        bool edges_connect(It_edge edge_a, It_edge edge_b) const;

        // Returns an edge that has it's internals swapped if it's flagged as inverted.
        //Edge normalized_edge(Index e) const; 
        //static Edge normalized_edge(const std::vector<Edge>& edges, Index e);

        //
        It_edge find_edge(It_vertex vertex_a, It_vertex vertex_b);

        // Calculates the mid point for face.
        glm::vec3 face_point(const Face&) const;
        //glm::vec3 face_point(It_face f) const { return face_point(*f); }

        // Calculates the mid point of edge.
        glm::vec3 edge_point(It_edge_const edge) const;
        //glm::vec3 edge_point(Index e) const { return edge_point(edges_[edge_index(e)]); } 

        //
        /*template <typename F>
        void for_each_face_sharing_vertex(It_vertex v, F&& f) const {
            auto cur_face = next_face(v);
            const auto end_face = cur_face; 
            while (cur_face != Invalid_index) {
                f(cur_face);
                cur_face = next_face(v, cur_face); 
                if (cur_face == end_face) {
                    cur_face = Invalid_index;
                }
            }
        }*/

        // Traverse each (normalized)edge connected to supplied vertex in clock-wise order.
        // Starting edge is "random", depending on construction order.
        template <typename F>
        void for_each_edge(It_vertex v, F&& f) const {

            auto edge_begin = v->edges.front();
            auto edge_cur = edge_begin;

            for (;;) { 
                f(edge_cur); 
                auto edge = edge_cur->aligned_edge(v); 
                auto right_face = edge.faces[SIDE_RIGHT]; 
                edge_cur = right_face->prev_edge(edge_cur);
                if (edge_cur == edge_begin) {
                    break;
                } 
            }
        }

        //
        /*std::vector<It_edge> get_edge_fan(It_vertex v) const
        {
            std::vector<It_edge> result;
            for_each_edge(v, [this, v, &result] (It_edge e) {
                result.push_back(e);
            });

            return result;
        }*/

        //
        /*std::vector<Edge> get_edge_fan_norm(It_vertex v, const std::vector<It_edge>& edge_fan) const
        {
            std::vector<Edge> result;

            std::transform(edge_fan.begin(), 
                           edge_fan.end(), 
                           std::back_inserter(result), 
                           [v] (It_edge e) {
                                return e->aligned_edge(v);
                            });

            return result;
        }*/

        //
        /*std::vector<Edge> get_edge_fan_norm(It_vertex v) const
        {
            auto indices = get_edge_fan(v);
            return get_edge_fan_norm(v, indices);
        }*/

        //
        /*std::vector<It_face> get_face_fan(It_vertex v) const
        {
            auto edges = get_edge_fan_norm(v);

            std::vector<It_face> result;

            for (const auto& edge : edges) {
                result.push_back(edge.faces[SIDE_RIGHT]);
            }

            return result;
        }*/


        // Pinches specified edge, which results in it being removed. 
        // First vertex is removed and any edges connecting to it will instead connect to second vertex.
        // Faces may be removed along with edges that become perpendicular. (Face becomes a straight line)
        //
        // Before:          After:
        //
        // \   /            \   /
        //  \ /              \ /
        //   .VK              . VK
        //   | E             / \
        //   .              /   \
        //  / \
        // /   \
        //
        // E = Edge to be removed(pinched)
        // VK = Vertex to keep
        //
        void pinch_edge(It_edge edge, It_vertex vertex_keep);

    public :
        // Tests if all edges connects to a face on each side.
        bool test_edge_integrity() const;

        // Verify that some assumptions about a face, with edges etc., holds true. 
        bool test_face_edge_integrity() const;
        bool test_face_edge_integrity(It_face face) const;

        // Verify that edges in a face is in clock-wise order.
        bool test_edge_order() const;

    public :
        glm::vec3 calc_face_normal(const Face& face) const;
    };

    /*Mesh mesh;
    transform::rotate(mesh, );
    transform::scale(mesh, glm::vec3(1.0, 1.0, 2.0));

    query::vertex_for(mesh, [&mesh] (auto it) {
    });*/

    struct query {

        template <typename F>
        static void vertex_for(Surface& s, F&& f) 
        {
            for (   auto it = s.vertices_.begin(); 
                    it != s.vertices_.end(); 
                    ++it) 
            {
                f(it);
            }
        }

        static bool edge_less(It_vertex v, It_edge edge_a, It_edge edge_b)
        {
            return edge_a->aligned_edge(v).faces[SIDE_RIGHT] == edge_b->aligned_edge(v).faces[SIDE_LEFT];
        }

        // Returns edges in clock-wise order.
        static std::vector<It_edge> edge_fan(const Surface& s, It_vertex vertex_it)
        {
            assert(vertex_it != s.vertices_.end());

            std::vector<It_edge> fan; 
            auto& edges = vertex_it->edges;
            fan.reserve(edges.size());
            if (edges.size() > 1) { 
                auto edge_it = edges.front();
                for (std::size_t i = 0; i < edges.size(); ++i) {
                    fan.push_back(edge_it); 
                    edge_it = edge_it->aligned_edge(vertex_it).faces[SIDE_RIGHT]->prev_edge(edge_it);
                }
            }
            return fan;
        }

        // Edges are in a clock-wise order and normalized relative 'vertex_it'.
        static std::vector<Edge> edge_fan_norm(const Surface& s, It_vertex vertex_it)
        {
            return edge_fan_norm(vertex_it, edge_fan(s, vertex_it));
        }

        //
        static std::vector<Edge> edge_fan_norm(It_vertex vertex_it, std::vector<It_edge> edges)
        {
            std::vector<Edge> fan;
            fan.reserve(edges.size());
            for (auto& edge : edges) {
                fan.push_back(edge->aligned_edge(vertex_it));
            }
            return fan; 
        }

    };

} // namespace ktool
