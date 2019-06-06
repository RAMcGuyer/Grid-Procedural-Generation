#include "Coord2D.h"

std::ostream& operator <<(std::ostream& out, const Coord2D& printMe) {

    out << "(" << printMe.first << ", " << printMe.second << ")";
    return out;
}

std::string coord_to_string(const Coord2D& printMe) {
    return "(" + std::to_string(printMe.first) + ", " + std::to_string(printMe.second) + ")";
}
