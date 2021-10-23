#pragma once

class Position
{
public:
    Position(void) {}

private:

};

using uptrPositions = std::unique_ptr<std::vector<Position>>;
