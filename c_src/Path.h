#ifndef __PATH_H__
#define __PATH_H__


#include <iostream>
#include <list>
#include <iterator>
#include <unordered_set>
#include <climits>

#include "Grid2D.h"
#include "Tile.h"
#include "Hash.h"

class Path {
    private:
        Grid2D* grid;
        std::list<Coord2D>* joints;
        int thickness;

    public:
        Path(Grid2D* grid);
        Path(Grid2D* grid, Coord2D point1, Coord2D point2, int thickness);
        Path(Grid2D* grid, std::list<Coord2D> & joints, int thickness);
        ~Path();

        bool areCompatibleJoints(Coord2D joint1, Coord2D joint2);
        bool addJoint(Coord2D newJoint);
        bool addJoint(Coord2D newJoint, int index);
        void setPathType(Tile::TileType type, bool prioritize);
        void populateBestPath(Coord2D src, Coord2D dest);
};


#endif // __PATH_H__
