#include <iostream>
#include <vector>
#include <string>

using namespace std;

#include "matching_engine.h"

int main()
{
  std::cout << "Matching engine test" << std::endl;

  MatchingEngine matching_engine;
  matching_engine.action(BuySell::buy);

  return 0;
}
