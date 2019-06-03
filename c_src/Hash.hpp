#ifndef __HASH_HPP__
#define __HASH_HPP__

#include <iostream>
#include "Tile.hpp"
#include "Coord2D.hpp"

struct TileHasher {
    size_t operator()(const Tile& tileObj) const {
		return std::hash<int>()(tileObj.getLocation()->hashCode());
	}
};

struct TilePtrHasher {
    size_t operator()(Tile* tileObj) const {
		return std::hash<int>()(tileObj->getLocation()->hashCode());
	}
};

struct TileComparator {
    bool operator()(const Tile& tileObj1, const Tile& tileObj2) const {
			if(tileObj1.getLocation() == 0) {
				std::cout<<"1"<<std::endl;
				std::cout<<tileObj1.toString()<<std::endl;
				exit(1);
			}
			if(tileObj2.getLocation() == 0) {
				std::cout<<"2"<<std::endl;
				std::cout<<tileObj2.toString()<<std::endl;
				exit(1);
			}
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

struct TilePtrComparator {
    bool operator()(Tile* tileObj1, Tile* tileObj2) const {
			if(tileObj1->getLocation() == 0) {
				std::cout<<"1"<<std::endl;
				std::cout<<tileObj1->toString()<<std::endl;
				exit(1);
			}
			if(tileObj2->getLocation() == 0) {
				std::cout<<"2"<<std::endl;
				std::cout<<tileObj2->toString()<<std::endl;
				exit(1);
			}
		if(
			tileObj1->getLocation()->getX() == tileObj2->getLocation()->getX()
			&& tileObj1->getLocation()->getY() == tileObj2->getLocation()->getY()
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

