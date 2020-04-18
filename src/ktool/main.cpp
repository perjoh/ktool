#include "surface.hpp"
#include "surface_io.hpp"
#include "argparser.hpp"
#include <klib/file_io.hpp>
#include <kgfx/opengl/renderer.hpp>
#include <kgfx/opengl/shader.hpp>
#include <kgfx/mesh.hpp>
#include <kgfx/event_handler.hpp>
#include <iostream>
#include <stdexcept>

//
ktool::Surface read_surface_stdin()
{
    auto opt_surface = ktool::io::read_surface(std::cin);
    if (!opt_surface.has_value()) {
        throw std::runtime_error("Failed to read surface from stdin.");
    }
    return opt_surface.value_or(ktool::Surface());
}

//
class Surface_renderer : public kgfx::Render_system
{
public : 
    Surface_renderer()
    {
        //
        auto surface = read_surface_stdin();
        surface.to_triangles(); 
        auto mesh = import_surface(surface);
        mesh.set_color(glm::vec3(1.0, 0.0, 0.0));
        mesh_.load_mesh(mesh);

        //
        auto vertex_shader = kgfx::opengl::Shader::load_from_source_file("basic.vs.glsl");
        auto fragment_shader = kgfx::opengl::Shader::load_from_source_file("basic.fs.glsl");
        kgfx::opengl::Shader_program shader(vertex_shader, fragment_shader);
        shader_.swap(shader);

        mvp_ = shader_.get_uniform<glm::mat4>("model_view_projection");
        model_transform_ = shader_.get_uniform<glm::mat3>("model_transform");
    }

private : 
    static kgfx::Triangle_mesh<> import_surface(const ktool::Surface& surface)
    {
        auto raw = surface.export_raw();

        const auto& indices = std::get<1>(raw);
        const auto& vertices = std::get<2>(raw);

        auto mesh = kgfx::Triangle_mesh<>::import_raw(&vertices[0], 
                                                     vertices.size(), 
                                                     &indices[0], 
                                                     indices.size()); 
        mesh.make_non_indexed();
        mesh.calculate_vertex_normals();
        return mesh;
    }

    void update(kgfx::Frame_time& frame_time) override
    { 
        static double angle = 0.0;
        const double delta = frame_time.delta_time_sec()*0.01*2.0*glm::pi<double>();
        angle += delta;

        camera_pos_ = glm::vec3(glm::cos(angle)*25.0, 
                                10.0, //glm::sin(angle)*25.0 + 25.0, 
                                glm::sin(angle)*25.0);
    }

    void render() override
    {
        auto shader_scope = shader_.bind_scope();

        const glm::mat4 projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);

        const glm::mat4 view = glm::lookAt(camera_pos_, //glm::vec3(5, 50, 10),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)); // *(glm::mat4(1.0) * glm::vec4(-1.0, 1.0, 1.0, 1.0));
        
        const glm::mat4 model_transform = glm::scale(glm::mat4(1.0), glm::vec3(5.0, 5.0, 5.0));

        mvp_.set(projection*view*model_transform);
        model_transform_.set(glm::mat3(1.0));

        mesh_.render();
    }

private :
    kgfx::opengl::Mesh mesh_;

    kgfx::opengl::Shader_program shader_;
    kgfx::opengl::Shader_uniform<glm::mat4> mvp_;
    kgfx::opengl::Shader_uniform<glm::mat3> model_transform_; // For transforming the normals. (Transform light vector instead?)

    glm::vec3 camera_pos_;
};

using Arguments = ktool::args::Parser::Arguments;

//
void command_cube(const Arguments& args)
{
    auto cube = ktool::Surface::construct_cube();

    if (args.size() > 0) {
        if (args.size() == 1) {
            cube.scale(glm::vec3(args[0].value<float>(1.0)));
        } else if (args.size() == 3) {
            cube.scale(glm::vec3(args[0].value<float>(1.0),
                                 args[1].value<float>(1.0),
                                 args[2].value<float>(1.0)));
        }
    }

    ktool::io::write_surface(cube, std::cout);
}


//
void command_offset(const Arguments& args)
{
    if (args.size() == 3) {
        auto surface = read_surface_stdin();
        surface.offset(glm::vec3(args[0].value<float>(0),
                                 args[1].value<float>(0),
                                 args[2].value<float>(0)));

        ktool::io::write_surface(surface, std::cout);
    }
}

//
void command_scale(const Arguments& args)
{
    if (args.size() == 3) {
        auto surface = read_surface_stdin();
        surface.scale(glm::vec3(args[0].value<float>(0),
                                args[1].value<float>(0),
                                args[2].value<float>(0)));

        ktool::io::write_surface(surface, std::cout);
    }
}

//
void command_subdivide(const Arguments& args)
{
    auto surface = read_surface_stdin();
    int count = 1;
    if (args.size() > 0) {
        count = std::min(args[0].value<int>(1), 3);
    }
    for (int i = 0; i < count; ++i) {
        surface.sub_divide();
    }

    ktool::io::write_surface(surface, std::cout);
}

//
void command_average(const Arguments& args)
{
    auto surface = read_surface_stdin(); 
    int count = 1;
    if (args.size() > 0) {
        count = args[0].value<int>(1);
    }

    for (int i = 0; i < count; ++i) {
        surface.average();
    }
    ktool::io::write_surface(surface, std::cout);
}

//
void command_view(const Arguments&)
{
    kgfx::Event_handler event_handler;

    kgfx::opengl::Renderer renderer;
    renderer.construct_windowed(800, 600, "ktool viewer");

    Surface_renderer surface_renderer;
    renderer.add_render_system(&surface_renderer);

    // 
    while (event_handler.poll_events()) {
        renderer.run_frame();
    }
}

//
int main(int argc, char* argv[])
{
    try
    {
        ktool::args::Parser parser("ktool v0.01", ktool::args::from_argv(argc, argv));

        parser.register_command("cube", &command_cube, "[size] Construct a cube.");
        parser.register_command("offset", &command_offset, "[x y z] Offset surface by xyz.");
        parser.register_command("scale", &command_scale, "[x y z] Scale surface by xyz.");
        parser.register_command("subdiv", &command_subdivide, "[count] Sub divide surface by count times.");
        parser.register_command("avg", &command_average, "Average(smooth) surface.");
        parser.register_command("view", &command_view, "Render surface.");

        parser.run_command(); 
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
