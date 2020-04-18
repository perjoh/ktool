#include <catch.hpp>
#include "argparser.hpp"

TEST_CASE("", "[argparser]")
{
    char* argv[3] = {"ktool.exe", "cube", "1.25"};

    const auto args = ktool::args::from_argv(3, argv);
    REQUIRE(args.size() == 3);
    REQUIRE(args[0] == argv[0]);
    REQUIRE(args[1] == argv[1]);
    REQUIRE(args[2] == argv[2]);

    ktool::args::Parser parser("", ktool::args::from_argv(3, argv));
    parser.register_command("cube", [](const ktool::args::Parser::Arguments& args){
        REQUIRE(args.size() == 1);
        REQUIRE(args[0].value<float>() == Approx(1.25f));
    });
    parser.run_command();
}
