#include <iostream>
#include <algorithm>
#ifdef RICH_MPI
  #include <mpi.h>
#endif // RICH_MPI
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
  std::string prefix = "";
  #ifdef RICH_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    prefix = "============" + to_string(rank) + "============ ";
  #endif // RICH_MPI
  os.precision(14);
  os << prefix << eo.getErrorMessage() << std::endl;
  for_each(eo.fields_.begin(), eo.fields_.end(),
          [&os, &prefix](const pair<string, UniversalError::PrintableAny>& f) {os << prefix << f.first << " " << f.second << endl;});
}