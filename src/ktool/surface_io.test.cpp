#include "surface_io.hpp"
#include <catch.hpp>
#include <sstream>
#include <limits>
#include <glm/gtc/epsilon.hpp>

TEST_CASE("Integrity test", "[surface_io]")
{
    auto cube = ktool::Surface::construct_cube();
    std::stringstream ss;
    ktool::io::write_surface(cube, ss);
    auto result = ktool::io::read_surface(ss);

    auto vec_a = glm::vec3(1.0, 2.0, 3.0);
    const auto eps = std::numeric_limits<glm::vec3::value_type>::epsilon();
    REQUIRE(glm::all(glm::epsilonEqual(vec_a, vec_a, eps)));
    REQUIRE(!glm::all(glm::epsilonEqual(vec_a, glm::vec3(1.1, 2.1, 3.1), eps)));
    REQUIRE(result.has_value());
    //REQUIRE(cube == result.value());
}

