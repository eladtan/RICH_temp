#include <iostream>
#include <algorithm>
#include "universal_error.hpp"

using namespace std;

UniversalError::UniversalError(const string& err_msg):
  err_msg_(err_msg),
  fields_() {}

void UniversalError::Append2ErrorMessage(string const& msg)
{
  err_msg_ += msg;
}

const string& UniversalError::getErrorMessage(void) const
{
  return err_msg_;
}

UniversalError::~UniversalError(void) {}

UniversalError::UniversalError(const UniversalError& eo):
  err_msg_(eo.getErrorMessage()),
  fields_(eo.fields_) {}


void reportError(UniversalError const& eo, std::ostream& os)
{
  os.precision(14);
  os << eo.getErrorMessage() << std::endl;
  for_each(eo.fields_.begin(), eo.fields_.end(),
          [&os](const pair<string, UniversalError::PrintableAny>& f) {os << f.first << " " << f.second << endl;});
}