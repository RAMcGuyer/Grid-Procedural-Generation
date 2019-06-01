#ifndef __HASH_HPP__
#define __HASH_HPP__

#include "Tile.hpp"
#include "Coord2D.hpp"

struct TileHasher {
    size_t operator()(const Tile& tileObj) const {
		return std::hash<int>()(tileObj.getLocation()->hashCode());
	}
};

struct TileComparator {
    bool operator()(const Tile& tileObj1, const Tile& tileObj2) const {
		if(
			tileObj1.getLocation()->getX() == tileObj2.getLocation()->getX()
			&& tileObj1.getLocation()->getY() == tileObj2.getLocation()->getY()
		) {
			return true;
		}
		else {
			return false;
		}
	}
};

struct Coord2DHasher {
	size_t operator()(const Coord2D& coord2D) const {
		return std::hash<int>()(coord2D.hashCode());
	}
};

struct Coord2DComparator {
	bool operator()(const Coord2D& obj1, const Coord2D& obj2) const {
		if(
			obj1.getX() == obj2.getX()
			&& obj1.getY() == obj2.getY()
		) {
			return true;
		}
		return false;
	}
};

#endif //__HASH_HPP__

