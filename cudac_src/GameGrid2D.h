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
//#include "Hash.h"

class GameGrid2D : public Grid2D {
    public:
        void AllocateAndCall(vector<Path> &paths, Coord2D* srcs, Coord2D* dests,  int path_sz);
        int determineSize(Coord2D c1, Coord2D c2);
        void swapSrc(Coord2D* src, Coord2D* dest);
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
    	std::list<Coord2D> getDistinctRandomPoints(unsigned int amount, std::set<Coord2D>&pointsSet);	
    	std::vector<Path> getFullPath(std::list<Coord2D> landmarks, int thickness);
    	Coord2D getRandomNonBase(); 
        
};    


#endif //__GAMEGRID2D_H__
