#ifndef __TILEHASH_H__
#define __TILEHASH_H__

#include "Tile.hpp"
#include "Coord2D.hpp"

struct TileHasher;
struct TileComparator;

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
#endif