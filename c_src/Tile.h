#ifndef __TILE_H__
#define __TILE_H__


#include <climits>
#include <string>

#include "Coord2D.h"

class Tile {
    public:
        enum TileType{
                EMPTY,
                TRAVERSABLE,
                NON_TRAVERSABLE
            };
        
    public:
        Tile();
        Tile(TileType type);
        Tile(TileType type,Coord2D location);
        Tile(TileType type, bool mark);
        Tile(const Tile& t);
        ~Tile();
        std::string getChar() const;
        Tile::TileType getType() const;
        bool isMarked() const;
        void setMark(bool mark);
        void setType(TileType t);
        std::string toString() const;
        void setDistance(int distance);
        int getDistance() const;
        void setPreviousTile(Tile* prev);
        Tile* getPreviousTile() const;
        Coord2D* getLocation() const;

    private:
        bool marked;
        TileType type;
        int distance;
        Tile* prev; 
        Coord2D* location;
};


#endif //__TILE_H__
