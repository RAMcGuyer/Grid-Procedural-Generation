#ifndef __PATH_H__
#define __PATH_H__


#include <iostream>
#include <list>
#include <iterator>
#include <unordered_set>
#include <climits>

#include "Grid2D.h"
#include "Tile.h"
//#include "Hash.h"

class Path {
    private:
        //Empty for now because it was giving me issues.
    public:
        Grid2D* grid;
        //std::list<Coord2D *> * joints;
        std::list<Coord2D>* joints;
        int thickness;
        Coord2D src;
        Coord2D dst;
        Path(Grid2D* grid);
        Path(Grid2D* grid, Coord2D src, Coord2D dst, int thickness);
        Path(Grid2D* grid, std::list<Coord2D> & joints, int thickness);
		Path(const Path& other);
        ~Path();

        bool areCompatibleJoints(Coord2D joint1, Coord2D joint2);
        bool addJoint(Coord2D newJoint);
        bool addJoint(Coord2D newJoint, int index);
        void setPathType(Tile::TileType type, bool prioritize);
};


#endif // __PATH_H__
