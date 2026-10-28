#ifndef ENVIRONMENT_PARSER_HPP
#define ENVIRONMENT_PARSER_HPP 1

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ArgumentParser.hpp"

// The environment variables a program reads, declared like its command-line arguments:
//
//   EnvironmentParser environment("Mach2 benchmark");
//   environment.addRequired<std::string>("RICH_OUTPUT_DIR", "directory for the results");
//   environment.addOptional<int>("STORM_SEED", 7, "Monte Carlo seed");
//   environment.parse();   // throws, naming every missing or malformed variable
//   const std::string dir = environment.get<std::string>("RICH_OUTPUT_DIR");
//
// Values parse as ArgumentParser's do. An empty variable counts as unset.
// submit/rich-submit reads these declarations from a benchmark's sources (and the runs/
// headers it includes) to show the variables and to require the required ones, so keep
// the name and the description as string literals.
class EnvironmentParser
{
private:
    struct VariableBase
    {
        VariableBase(const std::string &name, const std::string &description, bool required)
            : name_(name),
              description_(description),
              required_(required),
              wasSet_(false)
        {}

        virtual ~VariableBase()
        {}

        virtual void SetFromString(const std::string &text) = 0;
        virtual void Reset() = 0;
        virtual std::string DefaultAsString() const = 0;

        std::string name_;
        std::string description_;
        bool required_;
        bool wasSet_;
    };

    template<typename T>
    struct Variable : VariableBase
    {
        Variable(const std::string &name, const std::string &description, bool required, const T &defaultValue)
            : VariableBase(name, description, required),
              value_(defaultValue),
              defaultValue_(defaultValue)
        {}

        void SetFromString(const std::string &text) override
        {
            value_ = argument_parser_detail::ValueTraits<T>::Parse(text);
            wasSet_ = true;
        }

        void Reset() override
        {
            value_ = defaultValue_;
            wasSet_ = false;
        }

        std::string DefaultAsString() const override
        {
            return argument_parser_detail::ValueTraits<T>::ToString(defaultValue_);
        }

        T value_;
        T defaultValue_;
    };

public:
    explicit EnvironmentParser(const std::string &description = "")
        : description_(description),
          variables_(),
          variablesByName_()
    {}

    template<typename T>
    void addRequired(const std::string &name, const std::string &description = "")
    {
        Add(std::unique_ptr<VariableBase>(new Variable<T>(name, description, true, T())));
    }

    template<typename T>
    void addOptional(const std::string &name, const T &defaultValue, const std::string &description = "")
    {
        Add(std::unique_ptr<VariableBase>(new Variable<T>(name, description, false, defaultValue)));
    }

    // Reads every declared variable; throws std::runtime_error listing all the problems at once
    void parse()
    {
        std::vector<std::string> problems;
        for(auto &variable : variables_)
        {
            variable->Reset();
            const char *text = std::getenv(variable->name_.c_str());
            if(text == nullptr || text[0] == '\0')
            {
                if(variable->required_)
                    problems.push_back(variable->name_ + " is not set" +
                                       (variable->description_.empty() ? "" : " (" + variable->description_ + ")"));
                continue;
            }
            try
            {
                variable->SetFromString(text);
            }
            catch(const std::exception &error)
            {
                problems.push_back(variable->name_ + "='" + text + "' is not valid: " + error.what());
            }
        }
        if(!problems.empty())
        {
            std::ostringstream message;
            message << "Environment variables:";
            for(const auto &problem : problems)
                message << "\n  " << problem;
            message << "\n\n" << help();
            throw std::runtime_error(message.str());
        }
    }

    template<typename T>
    const T &get(const std::string &name) const
    {
        const Variable<T> *variable = dynamic_cast<const Variable<T>*>(Find(name));
        if(variable == nullptr)
            throw std::logic_error("Environment variable has different type: " + name);
        return variable->value_;
    }

    bool wasSet(const std::string &name) const
    {
        return Find(name)->wasSet_;
    }

    std::string help() const
    {
        std::ostringstream stream;
        if(!description_.empty())
            stream << description_ << '\n';
        stream << "Environment variables:\n";
        for(const auto &variable : variables_)
        {
            stream << "  " << variable->name_ << '\n';
            if(!variable->description_.empty())
                stream << "      " << variable->description_ << '\n';
            if(variable->required_)
                stream << "      required\n";
            else
                stream << "      default: " << variable->DefaultAsString() << '\n';
        }
        return stream.str();
    }

private:
    void Add(std::unique_ptr<VariableBase> variable)
    {
        if(variablesByName_.count(variable->name_) != 0)
            throw std::logic_error("Environment variable declared twice: " + variable->name_);
        variablesByName_[variable->name_] = variable.get();
        variables_.push_back(std::move(variable));
    }

    const VariableBase *Find(const std::string &name) const
    {
        auto found = variablesByName_.find(name);
        if(found == variablesByName_.end())
            throw std::logic_error("unknown environment variable: " + name);
        return found->second;
    }

    std::string description_;
    std::vector<std::unique_ptr<VariableBase>> variables_;
    std::unordered_map<std::string, VariableBase*> variablesByName_;
};

#endif // ENVIRONMENT_PARSER_HPP
