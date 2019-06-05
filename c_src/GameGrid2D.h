#ifndef __GAMEGRID2D_H__
#define __GAMEGRID2D_H__


#include <bits/stdc++.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <set>
#include <vector>

using namespace std;
using namespace std::chrono;

#include "Path.h"
#include "Coord2D.h"
#include "Grid2D.h"
#include "Hash.h"

class GameGrid2D : public Grid2D {
    public:
        GameGrid2D(Coord2D dimensions, int thickness, int landmarks) : Grid2D::Grid2D(dimensions) {
            init(thickness, landmarks);
        }

        GameGrid2D(const Grid2D other) : Grid2D::Grid2D(other) {
            init(3, 6);
        }

        ~GameGrid2D() {
            //This is a deconstructor 
        }
    private: 
        const int BASE_WIDTH = 8;
        Coord2D p1UpRight;
        Coord2D p2LowLeft;

    	void drawBases();
    	void init(int thickness, int numLandmarks);
    	std::list<Coord2D> getDistinctRandomPoints(unsigned int amount, std::unordered_set<Coord2D,Coord2DHasher,Coord2DComparator>&pointsSet);	
    	std::vector<Path> getFullPath(std::list<Coord2D> landmarks, int thickness);
    	Coord2D getRandomNonBase(); 
        void populateBestPath(Path p, Coord2D src, Coord2D dest);
};    


#endif //__GAMEGRID2D_H__
