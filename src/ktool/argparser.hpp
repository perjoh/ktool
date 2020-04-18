#pragma once
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <tuple>

namespace ktool {
namespace args {

    std::vector<std::string> from_argv(int argc, char* argv[]); 

    /*
     * Render a simple cube on screen.
     *  $ ktool cube | ktool view
     *
     * Manipulate a shape and save to a file.
     *  $ ktool cube 1 | ktool move 0 0.5 0 | ktool scale 1 5 1 | ktool smooth 3 > smoothcube.json
     *
     * Render shape from disk.
     *  $ cat smoothcube.json | ktool view
     */

    class Parser
    {
    public :
        Parser(const std::string& program_title, std::vector<std::string> args);

    public :
        class Argument;
        using Arguments = std::vector<Argument>;
        using Command_handler = std::function<void(const Arguments&)>;

        void register_command(const std::string& command_name, Command_handler, const std::string& desc = "No description available.");
        void run_command();

    public :
        class Argument
        {
        private :
            friend class Parser;
            Argument()
            { }

            Argument(const char* arg)
                : arg_(arg)
            { }

            const char* arg_{nullptr};

        public :
            Argument(const Argument& rhs)
                : arg_(rhs.arg_)
            { }

        public :
            template <typename T>
            T value(T t_default = T()) const;
        };

    private :
        void print_help() const;

    private :
        std::string program_title_;
        std::vector<std::string> args_;

        using Command = std::tuple<Command_handler, std::string>;
        std::map<std::string, Command> command_map_;
    };

} // namespace args 
} // namespace ktool
