#ifndef __COORD2D_H__
#define __COORD2D_H__


#include <iostream>
#include <string>
#include <utility>

#define Coord2D std::pair<int, int>

//std::ostream& printCoord2D(std::ostream& out, const Coord2D& printMe);
std::ostream& operator <<(std::ostream& out, const Coord2D& printMe);
std::string coord_to_string(const Coord2D& printMe);

#endif // __COORD2D_H__
