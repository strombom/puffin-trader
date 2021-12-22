// TrainingData.cpp : Defines the entry point for the application.
//

#include "IntrinsicEvents.h"
#include "DelayedKlines.h"
#include "TrainingData.h"
#include "TickData.h"
#include "Symbols.h"


using namespace std;

int main()
{
	// https://public.bybit.com/trading/BTCUSDT/

	const auto symbol = string_to_symbol("BTCUSDT");
	auto tick_data = TickData{symbol};
	//tick_data.save_csv("E:/BitCounter/tick_data.csv");
	auto intrinsic_events = IntrinsicEvents{};
	intrinsic_events.calculate_and_save(symbol, tick_data);
	intrinsic_events.load(symbol);
	//intrinsic_events.save_csv("E:/BitCounter/intrinsic_events.csv");
	//auto intrinsic_events = IntrinsicEvents{ symbol };
	auto delayed_klines = DelayedKlines{ intrinsic_events, tick_data };

	return 0;
}
