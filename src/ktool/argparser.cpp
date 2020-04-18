#include "argparser.hpp"
#include <nlohmann/json.hpp>
#include <iostream>

namespace ktool {
namespace args {

    std::vector<std::string> from_argv(int argc, char* argv[])
    {
        return std::vector<std::string>(&argv[0], &argv[argc]);
    }

    Parser::Parser(const std::string& program_title, 
                   std::vector<std::string> args)
        : program_title_(program_title)
        , args_(args)
    {
    }

    void Parser::register_command(const std::string& command_name, 
                                  std::function<void(const Arguments&)> command_handler, 
                                  const std::string& desc)
    {
        command_map_[command_name] = std::make_tuple(command_handler, desc);
    }

    void Parser::run_command()
    {
        if (args_.size() > 1) {
            auto it = command_map_.find(args_[1]);
            if (it != command_map_.end()) {
                Arguments arguments;

                if (args_.size() > 2) {
                    for (auto arg = args_.begin() + 2; arg != args_.end(); ++arg) {
                        arguments.push_back(arg->c_str());
                    }
                }

                std::get<0>(it->second)(arguments);
            }
            else {
                std::cout << "Unknown command '" << args_[1] << "'.\n\n";

                print_help();
            }
        }
        else {
            print_help();
        }
    }

    void Parser::print_help() const
    {
        std::cout << " " << program_title_ << "\n\n";

        for (auto& p : command_map_) {
            std::cout << "  " << p.first << " - " << std::get<1>(p.second) << '\n';
        }
    }

    template <typename T>
    T from_str(const char* s);

    template <>
    std::string from_str(const char* s) {
        return s;
    }

    template <>
    float from_str(const char* s) {
        return std::stof(s);
    }

    template <>
    int from_str(const char* s) {
        return std::stoi(s);
    }

    template <typename T>
    T Parser::Argument::value(T t_default) const
    {
        if (arg_) {
            return from_str<T>(arg_);
        }
        return t_default;
    }

    template std::string Parser::Argument::value<std::string>(std::string) const;
    template float Parser::Argument::value<float>(float) const; 
    template int Parser::Argument::value<int>(int) const; 


} // namespace args
} // namespace ktool
