#pragma once
#include "surface.hpp"
#include <iostream>
#include <optional>

namespace ktool {
namespace io {

    std::optional<Surface> read_surface(std::istream&);

    void write_surface(const Surface&, std::ostream&);

} // namespace io
} // namespace ktool
