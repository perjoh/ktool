#include <catch.hpp>
#include "surface.hpp"

TEST_CASE("Constructed cube integrity tests.", "[surface]") 
{
    ktool::Surface cube = ktool::Surface::construct_cube();

    REQUIRE(cube.vertex_count() == 8);
    REQUIRE(cube.edge_count() == 12);
    REQUIRE(cube.face_count() == 6); 

    SECTION("Test edge integrity.") {
        REQUIRE(cube.test_edge_integrity());
    }

    SECTION("Test face edge integrity.") {
        REQUIRE(cube.test_face_edge_integrity());
    }

    SECTION("Test edge order.") {
        REQUIRE(cube.test_edge_order());
    }

    SECTION("Test edge/face fanning.") {

        ktool::query::vertex_for(cube, [&cube] (auto vertex_it) {
            auto edges = ktool::query::edge_fan_norm(cube, vertex_it);
            for (auto& edge : edges) {
                REQUIRE(edge.vertices[0] == vertex_it);
            }

            REQUIRE(edges.size() == 3);
        });

        /*for (std::size_t i = 0; i < 8; ++i) {
            auto fan = cube.get_edge_fan_norm(i);
            auto num_edges_sharing_vertex = fan.size();
            REQUIRE(num_edges_sharing_vertex == 3);
            for (auto& edge : fan) {
                REQUIRE(edge.vertices[0] == i);
            }
        }

        for (std::size_t i = 0; i < 8; ++i) {
            auto fan = cube.get_face_fan(i);
            auto num_faces_sharing_vertex = fan.size();
            REQUIRE(num_faces_sharing_vertex == 3);
        }*/
    }
}

TEST_CASE("Export/import.", "[surface]")
{
    ktool::Surface cube = ktool::Surface::construct_cube();
    auto raw = cube.export_raw();

    REQUIRE(std::get<0>(raw).size() == 6);
    REQUIRE(std::get<1>(raw).size() == 6*4);
    REQUIRE(std::get<2>(raw).size() == 8*3);
}

TEST_CASE("Surface optimizing tests.", "[surface]")
{
    auto cube = ktool::Surface::construct_cube();
    auto cube2 = cube;

    cube.sub_divide();
    auto cube_sd = cube;
    cube_sd.to_triangles();
    //cube.optimize();

    //REQUIRE(cube == cube2);
}

/*TEST_CASE("Subdivided cube integrity tests.", "[surface]")
{
    ktool::Surface cube = ktool::Surface::construct_cube();

    cube.sub_divide(); 

    REQUIRE(cube.vertex_count() == 26); 
    REQUIRE(cube.edge_count() == 48);
    REQUIRE(cube.face_count() == 24); 

    SECTION("Test edge integrity.") {
        REQUIRE(cube.test_edge_integrity());
    }

    SECTION("Test face edge integrity.") {
        REQUIRE(cube.test_face_edge_integrity());
    }

    SECTION("Test edge order.") {
        REQUIRE(cube.test_edge_order());
    } 
}*/

TEST_CASE("Surfaces can be converted to triangles.", "[surface]")
{
    ktool::Surface cube = ktool::Surface::construct_cube();

    cube.to_triangles();

    REQUIRE(cube.vertex_count() == 8);
    REQUIRE(cube.edge_count() == 18);
    REQUIRE(cube.face_count() == 12);

    SECTION("Test edge integrity.") {
        REQUIRE(cube.test_edge_integrity());
    }

    SECTION("Test face edge integrity.") {
        REQUIRE(cube.test_face_edge_integrity());
    }

    SECTION("Test edge order.") {
        REQUIRE(cube.test_edge_order());
    }
}

