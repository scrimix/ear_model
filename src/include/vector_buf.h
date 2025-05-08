#pragma once

#include <ostream>
#include <vector>
#include <algorithm>

class VectorStreamBuf : public std::streambuf {
public:
  VectorStreamBuf(std::vector<unsigned char>& dest)
    : _dest(dest) {}

protected:
  // called when one character is written (e.g. os.put(ch) or os<<ch)
  int_type overflow(int_type ch) override {
    if (ch != traits_type::eof()) {
      _dest.push_back(static_cast<unsigned char>(ch));
      return ch;
    }
    return traits_type::eof();
  }

  // called when writing a sequence (e.g. os.write(buf,n) or os<<std::string)
  std::streamsize xsputn(const char* s, std::streamsize count) override {
    auto it = _dest.end();
    _dest.insert(it,
                 reinterpret_cast<const unsigned char*>(s),
                 reinterpret_cast<const unsigned char*>(s) + count);
    return count;
  }

private:
  std::vector<unsigned char>& _dest;
};
