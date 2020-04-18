#include "surface_io.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ktool {
namespace io {

/*

{
    "id" : "triangle",
    "polygons" : [3],
    "indices" : [0, 1, 2],
    "vertices" : [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
}

 */

    std::optional<Surface> read_surface(std::istream& is)
    {
        json j;
        is >> j;

        Surface::Raw_surface raw;

        const auto& polygons = j["polygons"];
        std::get<0>(raw).assign(polygons.begin(), polygons.end());

        const auto& indices = j["indices"];
        std::get<1>(raw).assign(indices.begin(), indices.end());

        const auto& vertices = j["vertices"];
        std::get<2>(raw).assign(vertices.begin(), vertices.end()); 

        return Surface::import_raw(raw);
    }

    void write_surface(const Surface& surface, std::ostream& os)
    {
        auto raw = surface.export_raw();

        json j;

        j["polygons"] = std::get<0>(raw);
        j["indices"] = std::get<1>(raw);
        j["vertices"] = std::get<2>(raw);

        os << j << '\n';
    }

} // namespace io
} // namespace ktool 
