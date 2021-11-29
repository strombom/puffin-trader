// TrainingData.cpp : Defines the entry point for the application.
//

#include "TrainingData.h"
#include "TickData.h"
#include "Symbols.h"


using namespace std;

int main()
{
	const auto symbol = string_to_symbol("BTCUSDT");
	auto tick_data = TickData{symbol};

	return 0;
}
