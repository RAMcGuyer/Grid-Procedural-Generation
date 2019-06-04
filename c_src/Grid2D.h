#ifndef __GRID2D_H__
#define __GRID2D_H__


#include <cassert>
#include <string>
#include <unordered_set>
#include <vector>
#include <iostream>

using namespace std;

#include "Tile.h"
#include "Coord2D.h"
#include "Hash.h"

class Grid2D {
	public:
		std::vector<std::vector<Tile*> >* grid;
		
		Grid2D();
		Grid2D(Coord2D dimensions);
		Grid2D(const Grid2D& other);
		Grid2D(Grid2D* other);
		~Grid2D();

		std::string toString();
		void setTile(Tile::TileType t, Coord2D location);
		Tile* getTile(Coord2D location) const;
		void assertBounds(Coord2D location) const;
		bool checkBounds(Coord2D location) const;
		Coord2D getGridDimensions();
		int size();
		std::string getChar(Coord2D location);
		bool canGoUp(Coord2D location);
		bool canGoDown(Coord2D location);
		bool canGoLeft(Coord2D location);
		bool canGoRight(Coord2D location);
		Tile* getUp(Coord2D fromHere);
		Tile* getDown(Coord2D fromHere);
		Tile* getLeft(Coord2D fromHere);
		Tile* getRight(Coord2D fromHere);
		void markLine(Coord2D point1, Coord2D point2, bool mark);
		void setTypeLine(Coord2D point1, Coord2D point2, Tile::TileType type, bool prioritize);
		void setTypeRect(Coord2D lowerLeft, Coord2D upperRight, Tile::TileType type, bool prioritize);
		void setTypeLine(Coord2D point1, Coord2D point2, Tile::TileType type, int layers, bool prioritize);
		void markRect(Coord2D lowerLeft, Coord2D upperRight, bool mark);
		std::unordered_set<Tile*, TilePtrHasher, TilePtrComparator> getTraversableNeighbors(Coord2D location);

	private:	
		int ROWS;
		int COLS;
};


#endif //__GRID2D_H__
