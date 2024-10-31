#ifndef TQA_SUBQUERY_HPP
#define TQA_SUBQUERY_HPP

namespace TQA
{
    template<typename QueryData>
    struct SubQueryData
    {
        size_t queryIndex;
        QueryData data;
    };
}

#endif // TQA_SUBQUERY_HPP