#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP 1

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace argument_parser_detail
{

inline bool StartsWithDash(const std::string &value)
{
    return !value.empty() && value[0] == '-';
}

inline std::string StripLeadingDashes(const std::string &name)
{
    size_t pos = 0;
    while(pos < name.size() && name[pos] == '-')
        pos++;
    return name.substr(pos);
}

inline std::string OptionToken(const std::string &name)
{
    if(StartsWithDash(name))
        return name;
    return "--" + name;
}

template<typename T>
typename std::enable_if<std::is_unsigned<T>::value, void>::type
RejectNegativeUnsigned(const std::string &text)
{
    if(!text.empty() && text[0] == '-')
        throw std::invalid_argument("negative value is not valid for an unsigned argument");
}

template<typename T>
typename std::enable_if<!std::is_unsigned<T>::value, void>::type
RejectNegativeUnsigned(const std::string&)
{}

template<typename T>
struct ValueTraits
{
    static T Parse(const std::string &text)
    {
        RejectNegativeUnsigned<T>(text);

        std::istringstream stream(text);
        T value;
        stream >> value;
        if(!stream || !(stream >> std::ws).eof())
            throw std::invalid_argument("could not parse value");
        return value;
    }

    static std::string ToString(const T &value)
    {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    }
};

template<>
struct ValueTraits<std::string>
{
    static std::string Parse(const std::string &text)
    {
        return text;
    }

    static std::string ToString(const std::string &value)
    {
        return value;
    }
};

template<>
struct ValueTraits<bool>
{
    static bool Parse(const std::string &text)
    {
        std::string lower(text);
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

        if(lower == "1" || lower == "true" || lower == "yes" || lower == "on")
            return true;
        if(lower == "0" || lower == "false" || lower == "no" || lower == "off")
            return false;

        throw std::invalid_argument("expected true/false");
    }

    static std::string ToString(bool value)
    {
        return value ? "true" : "false";
    }
};

} // namespace argument_parser_detail

class ArgumentParser
{
private:
    enum class ArgumentKind
    {
        Positional,
        Option,
        Flag
    };

    struct ArgumentBase
    {
        struct Alias
        {
            Alias(const std::string &token, bool implicitValue)
                : token(token),
                  implicitValue(implicitValue)
            {}

            std::string token;
            bool implicitValue;
        };

        ArgumentBase(ArgumentParser &owner,
                     ArgumentKind kind,
                     const std::string &name,
                     const std::string &description)
            : owner_(owner),
              kind_(kind),
              name_(name),
              description_(description),
              aliases_(),
              implicitValues_(),
              required_(false),
              hasDefault_(false),
              wasSet_(false),
              order_(0)
        {}

        virtual ~ArgumentBase()
        {}

        virtual void SetValueFromString(const std::string &text, size_t order) = 0;
        virtual void Reset() = 0;
        virtual std::string DefaultAsString() const = 0;
        virtual std::string ChoicesAsString() const = 0;
        virtual std::string ValueName() const = 0;
        virtual bool ExpectsValue() const = 0;

        std::string DisplayName() const
        {
            if(kind_ == ArgumentKind::Positional)
                return name_;
            return argument_parser_detail::OptionToken(name_);
        }

        ArgumentParser &owner_;
        ArgumentKind kind_;
        std::string name_;
        std::string description_;
        std::vector<Alias> aliases_;
        std::unordered_map<std::string, std::string> implicitValues_;
        bool required_;
        bool hasDefault_;
        bool wasSet_;
        size_t order_;
    };

public:
    template<typename T>
    class Argument : public ArgumentBase
    {
    public:
        Argument<T> &defaultValue(const T &value)
        {
            value_ = value;
            defaultValue_ = value;
            this->hasDefault_ = true;
            return *this;
        }

        Argument<T> &defaultsTo(const T &value)
        {
            return defaultValue(value);
        }

        Argument<T> &required(bool value = true)
        {
            this->required_ = value;
            return *this;
        }

        Argument<T> &mandatory(bool value = true)
        {
            return required(value);
        }

        Argument<T> &alias(const std::string &name)
        {
            if(this->kind_ == ArgumentKind::Positional)
                throw std::logic_error("positional arguments cannot have aliases");

            const std::string token = argument_parser_detail::OptionToken(name);
            this->owner_.RegisterOptionName(token, this);
            this->aliases_.push_back(typename ArgumentBase::Alias(token, false));
            return *this;
        }

        Argument<T> &optionAlias(const std::string &name)
        {
            const std::string token = argument_parser_detail::OptionToken(name);
            this->owner_.RegisterOptionName(token, this);
            this->aliases_.push_back(typename ArgumentBase::Alias(token, false));
            return *this;
        }

        Argument<T> &flagAlias(const std::string &name, const T &value)
        {
            const std::string token = argument_parser_detail::OptionToken(name);
            this->owner_.RegisterOptionName(token, this);
            this->aliases_.push_back(typename ArgumentBase::Alias(token, true));
            this->implicitValues_[token] =
                argument_parser_detail::ValueTraits<T>::ToString(value);
            return *this;
        }

        Argument<T> &choices(const std::vector<T> &values)
        {
            choices_ = values;
            return *this;
        }

        Argument<T> &choices(std::initializer_list<T> values)
        {
            choices_ = std::vector<T>(values);
            return *this;
        }

        const T &get() const
        {
            return value_;
        }

        operator const T&() const
        {
            return get();
        }

        bool wasSet() const
        {
            return this->wasSet_;
        }

        size_t order() const
        {
            return this->order_;
        }

    private:
        friend class ArgumentParser;

        Argument(ArgumentParser &owner,
                 ArgumentKind kind,
                 const std::string &name,
                 const std::string &description)
            : ArgumentBase(owner, kind, name, description),
              value_(),
              defaultValue_(),
              choices_()
        {}

        void SetValueFromString(const std::string &text, size_t order) override
        {
            T parsed;
            try
            {
                parsed = argument_parser_detail::ValueTraits<T>::Parse(text);
            }
            catch(const std::exception &e)
            {
                std::ostringstream message;
                message << "Invalid value for " << this->DisplayName()
                        << ": '" << text << "' (" << e.what() << ")";
                throw std::runtime_error(message.str());
            }

            if(!choices_.empty() &&
               std::find(choices_.begin(), choices_.end(), parsed) == choices_.end())
            {
                std::ostringstream message;
                message << "Invalid value for " << this->DisplayName()
                        << ": '" << text << "'. Expected one of: "
                        << ChoicesAsString();
                throw std::runtime_error(message.str());
            }

            value_ = parsed;
            this->wasSet_ = true;
            this->order_ = order;
        }

        void Reset() override
        {
            value_ = this->hasDefault_ ? defaultValue_ : T();
            this->wasSet_ = false;
            this->order_ = 0;
        }

        std::string DefaultAsString() const override
        {
            return argument_parser_detail::ValueTraits<T>::ToString(defaultValue_);
        }

        std::string ChoicesAsString() const override
        {
            std::ostringstream stream;
            for(size_t i = 0; i < choices_.size(); i++)
            {
                if(i > 0)
                    stream << ", ";
                stream << argument_parser_detail::ValueTraits<T>::ToString(choices_[i]);
            }
            return stream.str();
        }

        std::string ValueName() const override
        {
            std::string result = this->name_;
            std::transform(result.begin(), result.end(), result.begin(),
                           [](unsigned char c)
                           {
                               if(c == '-')
                                   return '_';
                               return static_cast<char>(std::toupper(c));
                           });
            return result;
        }

        bool ExpectsValue() const override
        {
            return this->kind_ != ArgumentKind::Flag;
        }

        T value_;
        T defaultValue_;
        std::vector<T> choices_;
    };

    explicit ArgumentParser(const std::string &description = "")
        : programName_("program"),
          description_(description),
          arguments_(),
          positionals_(),
          options_(),
          argumentsByName_(),
          optionsByToken_(),
          parseOrder_(0)
    {}

    template<typename T>
    Argument<T> &positional(const std::string &name, const std::string &description = "")
    {
        return AddArgument<T>(ArgumentKind::Positional,
                              argument_parser_detail::StripLeadingDashes(name),
                              description);
    }

    template<typename T>
    Argument<T> &positional(const std::string &name,
                            const T &defaultValue,
                            const std::string &description = "")
    {
        return positional<T>(name, description).defaultValue(defaultValue);
    }

    template<typename T>
    Argument<T> &addPositional(const std::string &name, const std::string &description = "")
    {
        return positional<T>(name, description);
    }

    template<typename T>
    Argument<T> &addPositional(const std::string &name,
                               const T &defaultValue,
                               const std::string &description = "")
    {
        return positional<T>(name, defaultValue, description);
    }

    template<typename T>
    Argument<T> &option(const std::string &name, const std::string &description = "")
    {
        return AddArgument<T>(ArgumentKind::Option,
                              argument_parser_detail::StripLeadingDashes(name),
                              description);
    }

    template<typename T>
    Argument<T> &option(const std::string &name,
                        const T &defaultValue,
                        const std::string &description = "")
    {
        return option<T>(name, description).defaultValue(defaultValue);
    }

    template<typename T>
    Argument<T> &addOption(const std::string &name, const std::string &description = "")
    {
        return option<T>(name, description);
    }

    template<typename T>
    Argument<T> &addOption(const std::string &name,
                           const T &defaultValue,
                           const std::string &description = "")
    {
        return option<T>(name, defaultValue, description);
    }

    Argument<bool> &flag(const std::string &name, const std::string &description = "")
    {
        return AddArgument<bool>(ArgumentKind::Flag,
                                 argument_parser_detail::StripLeadingDashes(name),
                                 description).defaultValue(false);
    }

    Argument<bool> &addFlag(const std::string &name, const std::string &description = "")
    {
        return flag(name, description);
    }

    bool parse(int argc, char *argv[])
    {
        if(argc > 0 && argv[0] != nullptr)
            programName_ = argv[0];

        for(auto &argument : arguments_)
            argument->Reset();

        parseOrder_ = 0;
        size_t positionalIndex = 0;
        bool namedArgumentsEnabled = true;

        for(int i = 1; i < argc; i++)
        {
            std::string token(argv[i]);

            if(namedArgumentsEnabled && token == "--")
            {
                namedArgumentsEnabled = false;
                continue;
            }

            if(namedArgumentsEnabled && (token == "--help" || token == "-h"))
                return false;

            if(namedArgumentsEnabled && argument_parser_detail::StartsWithDash(token))
            {
                std::string value;
                const size_t equalPos = token.find('=');
                if(equalPos != std::string::npos)
                {
                    value = token.substr(equalPos + 1);
                    token = token.substr(0, equalPos);
                }

                auto found = optionsByToken_.find(token);
                if(found == optionsByToken_.end())
                    throw std::runtime_error("Unknown argument: " + token);

                ArgumentBase *argument = found->second;
                parseOrder_++;

                auto implicitValue = argument->implicitValues_.find(token);
                if(implicitValue != argument->implicitValues_.end())
                {
                    if(equalPos != std::string::npos)
                        throw std::runtime_error(token + " does not take a value");
                    argument->SetValueFromString(implicitValue->second, parseOrder_);
                    continue;
                }

                if(argument->ExpectsValue())
                {
                    if(equalPos == std::string::npos)
                    {
                        if(i + 1 >= argc)
                            throw std::runtime_error("Missing value for " + token);
                        value = argv[++i];
                    }
                    argument->SetValueFromString(value, parseOrder_);
                }
                else
                {
                    argument->SetValueFromString(
                        equalPos == std::string::npos ? std::string("true") : value,
                        parseOrder_);
                }

                continue;
            }

            while(positionalIndex < positionals_.size() &&
                  positionals_[positionalIndex]->wasSet_)
            {
                positionalIndex++;
            }

            if(positionalIndex >= positionals_.size())
                throw std::runtime_error("Unexpected positional argument: " + token);

            parseOrder_++;
            positionals_[positionalIndex]->SetValueFromString(token, parseOrder_);
            positionalIndex++;
        }

        for(const auto &argument : arguments_)
        {
            if(argument->required_ && !argument->wasSet_)
                throw std::runtime_error("Missing required argument: " + argument->DisplayName());
        }

        return true;
    }

    std::string help() const
    {
        std::ostringstream stream;
        stream << "Usage: " << programName_;
        if(!options_.empty())
            stream << " [options]";
        for(const auto &argument : positionals_)
        {
            if(argument->required_)
                stream << " <" << argument->name_ << ">";
            else
                stream << " [" << argument->name_ << "]";
        }
        stream << '\n';

        if(!description_.empty())
            stream << '\n' << description_ << '\n';

        if(!positionals_.empty())
        {
            stream << "\nPositional arguments:\n";
            for(const auto &argument : positionals_)
                AppendArgumentHelp(stream, *argument, "  ");
        }

        stream << "\nOptions:\n";
        stream << "  -h, --help\n"
               << "      Show this help message.\n";
        for(const auto &argument : options_)
            AppendArgumentHelp(stream, *argument, "  ");

        return stream.str();
    }

    template<typename T>
    const T &get(const std::string &name) const
    {
        const ArgumentBase *base = FindByName(name);
        const Argument<T> *argument = dynamic_cast<const Argument<T>*>(base);
        if(argument == nullptr)
            throw std::logic_error("Argument has different type: " + name);
        return argument->get();
    }

    bool wasSet(const std::string &name) const
    {
        return FindByName(name)->wasSet_;
    }

private:
    template<typename T>
    Argument<T> &AddArgument(ArgumentKind kind,
                             const std::string &name,
                             const std::string &description)
    {
        if(name.empty())
            throw std::logic_error("argument name cannot be empty");
        if(argumentsByName_.find(name) != argumentsByName_.end())
            throw std::logic_error("duplicate argument name: " + name);

        std::unique_ptr<Argument<T>> argument(new Argument<T>(*this, kind, name, description));
        Argument<T> *raw = argument.get();

        arguments_.push_back(std::move(argument));
        argumentsByName_[name] = raw;

        if(kind == ArgumentKind::Positional)
        {
            positionals_.push_back(raw);
        }
        else
        {
            options_.push_back(raw);
            RegisterOptionName(raw->DisplayName(), raw);
        }

        return *raw;
    }

    void RegisterOptionName(const std::string &token, ArgumentBase *argument)
    {
        auto inserted = optionsByToken_.insert(std::make_pair(token, argument));
        if(!inserted.second && inserted.first->second != argument)
            throw std::logic_error("duplicate argument token: " + token);
    }

    const ArgumentBase *FindByName(const std::string &name) const
    {
        if(argument_parser_detail::StartsWithDash(name))
        {
            auto tokenFound = optionsByToken_.find(name);
            if(tokenFound != optionsByToken_.end())
                return tokenFound->second;
        }

        const std::string key = argument_parser_detail::StripLeadingDashes(name);
        auto found = argumentsByName_.find(key);
        if(found == argumentsByName_.end())
            throw std::logic_error("unknown argument name: " + name);
        return found->second;
    }

    static void AppendArgumentHelp(std::ostringstream &stream,
                                   const ArgumentBase &argument,
                                   const std::string &prefix)
    {
        stream << prefix << argument.DisplayName();
        if(argument.ExpectsValue())
            stream << " <" << argument.ValueName() << ">";
        for(const auto &alias : argument.aliases_)
        {
            stream << ", " << alias.token;
            if(argument.ExpectsValue() && !alias.implicitValue)
                stream << " <" << argument.ValueName() << ">";
        }
        stream << '\n';

        if(!argument.description_.empty())
            stream << prefix << "    " << argument.description_ << '\n';
        if(argument.required_)
            stream << prefix << "    required\n";
        if(argument.hasDefault_)
            stream << prefix << "    default: " << argument.DefaultAsString() << '\n';

        const std::string choices = argument.ChoicesAsString();
        if(!choices.empty())
            stream << prefix << "    choices: " << choices << '\n';
    }

    std::string programName_;
    std::string description_;
    std::vector<std::unique_ptr<ArgumentBase>> arguments_;
    std::vector<ArgumentBase*> positionals_;
    std::vector<ArgumentBase*> options_;
    std::unordered_map<std::string, ArgumentBase*> argumentsByName_;
    std::unordered_map<std::string, ArgumentBase*> optionsByToken_;
    size_t parseOrder_;
};

#endif // ARGUMENT_PARSER_HPP
