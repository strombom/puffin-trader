
#ifndef MATCHING_ENGINE__
#define MATCHING_ENGINE__

enum BuySell {
    buy  = 0,
    sell = 1
};

enum OrderType {
    limit              = 0,
    market             = 1,
    stop_market        = 2,
    stop_limit         = 3,
    trailing_stop      = 4,
    take_profit_limit  = 5,
    take_profit_market = 6
};

enum TriggerType {
    mark  = 0,
    last  = 1,
    index = 2
};

class MatchingEngine
{
    public:
        MatchingEngine(void);

        void action(BuySell buysell);
};


#endif
