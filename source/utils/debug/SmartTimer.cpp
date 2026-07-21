#include "SmartTimer.hpp"

#ifdef ALLOW_TIMING
namespace SmartTimer
{
    std::shared_ptr<Node> Node::MakeRoot(void)
    {
        std::shared_ptr<Node> root = std::make_shared<Node>();
        root->type = Node::FUNCTION;
        root->name = "main";
        return root;
    }

    std::shared_ptr<Node> Node::root = Node::MakeRoot();
    bool TimerCreator::globalSilent = false;
    bool TimerCreator::disable = false;

    std::vector<std::string> GetStack(void)
    {
        std::vector<std::string> out;
        const boost::stacktrace::stacktrace st;
        out.reserve(st.size());
        // Bottom to top (so "main" first)
        for(int i = static_cast<int>(st.size()) - 4; i >= 0; --i) 
        {
            // name() is usually enough; to_string(st[i]) can include file:line (needs -g)
            std::string name = st[i].name();
            name = name.substr(0, name.find("("));
            out.emplace_back(name);
        }

        out.resize(out.size() - 4);
        return out;
    }

    std::shared_ptr<Node> GetParentNode(bool preempt, bool verbose = false)
    {
        std::shared_ptr<Node> current = Node::root;
        std::vector<std::string> stack = GetStack();
        bool lastCreated;
        size_t i = 1;
        while(i < stack.size())
        {
            lastCreated = false;
            bool createNew = true;
            for(std::shared_ptr<Node> child : current->children)
            {
                if(child->type == Node::BLOCK and not child->timer->done)
                {
                    current = child;
                    createNew = false;
                    break;
                }
            }
            if(createNew)
            {
                std::string function = stack[i];
                if(current->children_function_map.find(function) == current->children_function_map.end())
                {
                    std::shared_ptr<Node> newNode = std::make_shared<Node>();
                    newNode->type = Node::FUNCTION;
                    newNode->name = function;
                    newNode->timer = std::make_shared<Timer>();
                    current->children_function_map[function] = newNode;
                    current->children.push_back(newNode);
                    current = newNode;
                    lastCreated = true;
                }
                else
                {
                    current = current->children_function_map[function];
                }
                i++;
            }
        }
        if(verbose)
        {
            std::cout << "Currently, creating under parent " << ((current->type == Node::Type::FUNCTION)? "Function " + current->name : " Scope " + current->name) << std::endl;
        }
        if(not lastCreated)
        {
            std::shared_ptr<Node> last = current;
            while(true)
            {
                bool found = false;
                for(std::shared_ptr<Node> child : current->children)
                {
                    if(child->type == Node::BLOCK and not child->timer->done)
                    {
                        last = current;
                        current = child;
                        found = true;
                        break;
                    }
                }
                if(not found)
                {
                    break;
                }
            }
            if(preempt)
            {
                if(current->timer)
                {
                    current->timer->Destroy();
                }
                current = last;
            }
        }
        return current;
    }

    std::shared_ptr<Node> GetNode(const std::string &name, bool distinct, bool preempt)
    {
        bool verbose = false;
        std::shared_ptr<Node> parent = GetParentNode(preempt, verbose);
        bool createNew = distinct;
        if(not createNew)
        {
            if(parent->children_block_map.find(name) == parent->children_block_map.end())
            {
                createNew = true;
            }
        }
        if(createNew)
        {
            std::shared_ptr<Node> newNode = std::make_shared<Node>();
            newNode->type = Node::BLOCK;
            newNode->name = name;
            newNode->timer = std::make_shared<Timer>();
            parent->children_block_map[name] = newNode;
            parent->children.push_back(newNode);
            if(verbose)
            {
                std::cout << "Created " << name << " under parent " << ((parent->type == Node::Type::FUNCTION)? "Function " + parent->name : " Scope " + parent->name) << std::endl;
            }
            return newNode;
        }
        else
        {
            return parent->children_block_map[name];
        }
    }

    TimerCreator::TimerCreator(const std::string &name, bool distinct, bool preempt)
    {
        if(not TimerCreator::disable)
        {
            this->silent = TimerCreator::globalSilent;
            if(not this->silent)
            {
                this->node = GetNode(name, distinct, preempt);
                std::shared_ptr<Node> node = this->node.lock();
                assert(node->timer != nullptr);
                node->creator = this;
                node->timer->done = false;
                node->timer->start = std::chrono::high_resolution_clock::now();
            }
            this->destroyed = false;
        }
    }

    void TimerCreator::Destroy(void)
    {
        if(TimerCreator::disable)
        {
            return;
        }
        if(this->destroyed)
        {
            return;
        }
        this->destroyed = true;
        if(not this->silent)
        {
            Node *node = this->node.lock().get();
            assert(node->timer != nullptr);
            if(node->timer)
            {
                node->timer->Destroy();
            }
            node->creator = nullptr;
        }
    }

    TimerCreator::~TimerCreator()
    {
        this->Destroy();
    }

    Node::~Node()
    {
        if(this->destroyed)
        {
            return;
        }
        this->destroyed = true;
        if(this->creator)
        {
            this->creator->destroyed = true;
        }
    }

    Timer::Timer()
    {
        if(not TimerCreator::disable)
        {
            #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
            #endif // RICH_MPI
            this->done = false;
            this->start = std::chrono::high_resolution_clock::now();
            this->time = 0;
            this->totalTime = 0;
        }
    }

    void Timer::Destroy(void)
    {
        if(TimerCreator::disable)
        {
            return;
        }
        if(std::uncaught_exceptions())
        {
            return;
        }
        if(not this->done)
        {
            #ifdef RICH_MPI
                MPI_Barrier(MPI_COMM_WORLD);
            #endif // RICH_MPI
            auto end = std::chrono::high_resolution_clock::now();
            // time in seconds
            this->time = std::chrono::duration_cast<std::chrono::duration<double>>(end - this->start).count();
            #ifdef RICH_MPI
                MPI_Allreduce(MPI_IN_PLACE, &this->time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            #endif // RICH_MPI
            this->totalTime += this->time;
            this->done = true;
        }
    }

    void Node::PrintHelper(std::ostream &stream, const std::vector<std::pair<size_t, std::string>> &indentations) const
    {
        for(size_t i = 1; i < indentations.size(); i++)
        {
            stream << indentations[i].second;
            for(size_t j = indentations[i-1].first; j < indentations[i].first; j++)
            {
                stream << " ";
            }
        }
        if(this->type == Node::Type::FUNCTION)
        {
            std::string color = (&stream == &std::cout)? "\033[0;34m" : "";
            std::string colorReset = (&stream == &std::cout)? "\033[0m" : "";
            stream << color << this->name << colorReset << std::endl;
        }
        if(this->type == Node::Type::BLOCK)
        {
            stream << this->name << ": " << this->timer->totalTime << std::endl;
        }
        std::vector<std::pair<size_t, std::string>> newIndentations;
        for(size_t i = 0; i < indentations.size(); i++)
        {
            newIndentations.emplace_back(std::pair<size_t, std::string>({indentations[i].first, "│"}));
        }
        for(size_t i = 0; i < this->children.size(); i++)
        {
            const auto &child = this->children[i];
            if(i == this->children.size() - 1 and child->children.empty())
            {
                newIndentations.emplace_back(std::make_pair<size_t, std::string>(indentations.back().first + 1, "└"));
            }
            else
            {
                newIndentations.emplace_back(std::make_pair<size_t, std::string>(indentations.back().first + 1, "├"));
            }
            child->PrintHelper(stream, newIndentations);
            newIndentations.pop_back();
        }
    }

    void PrintTimes(void)
    {
        if(TimerCreator::disable)
        {
            return;
        }

        Node::root->PrintHelper(std::cout, {{0, ""}});
    }
}
#endif // ALLOW_TIMING