/*! \file universal_error.hpp
  \brief A class for storing error and debug information
  \author Almog Yalinewich, Maor Mizrachi
 */

#ifndef UNIVERSAL_ERROR_HPP
#define UNIVERSAL_ERROR_HPP 1

#include <iostream>
#include <string>
#include <vector>
#include <any>
#include "utils/printing/print.hpp"

using std::string;
using std::vector;
using std::pair;

/*! \brief Container for error reports
 */
class UniversalError
{
private:

  struct PrintableAny
  {
    using PrintFunction = void(*)(std::ostream&, const std::any&);

    template<typename T>
    inline PrintableAny(const T &value)
    {
      this->value_ = value;
      this->printFunction_ = [](std::ostream &os, const std::any &value)
                              {
                                os << std::any_cast<T>(value);
                              };
    }

    inline friend std::ostream &operator<<(std::ostream &os, const PrintableAny &p)
    {
      p.printFunction_(os, p.value_);
      return os;
    }

    std::any value_;
    PrintFunction printFunction_;
  };

public:

  /*! \brief Class constructor
    \param err_msg Error message
   */
  explicit UniversalError(const string& err_msg);

  /*! \brief Appends std::string to the error message
    \param msg Message to append
   */
  void Append2ErrorMessage(std::string const& msg);

  /*! \brief Returns the error message
    \return Error message
   */
  std::string const& getErrorMessage(void) const;

  ~UniversalError(void);

  /*! \brief Copy constructor
    \param eo Source
   */
  UniversalError(const UniversalError& eo);

  template<typename T>
  UniversalError& addEntry(const std::string &name, const T &value)
  {
    this->fields_.emplace_back(name, value);
    return *this;
  }

  /*! \brief Prints the contents of the error
  \param eo The error object
  */
  friend void reportError(UniversalError const& eo, std::ostream& os);

private:

  string err_msg_;

  std::vector<std::pair<std::string, PrintableAny>> fields_;
};

void reportError(UniversalError const& eo, std::ostream& os = std::cout);

#endif // UNIVERSAL_ERROR_HPP
