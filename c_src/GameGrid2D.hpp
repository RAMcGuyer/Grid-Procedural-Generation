#ifndef __GAMEGRID2D_HPP__
#define __GAMEGRID2D_HPP__

#include <bits/stdc++.h>
#include <stdlib.h>
#include <time.h>
#include <set>
#include <vector>

#include "Path.hpp"
#include "Coord2D.hpp"
#include "Grid2D.hpp"
#include "Hash.hpp"

using namespace std;

class GameGrid2D : public Grid2D {
public:

    // vector<Path> getFullPath(list<Coord2D> landmarks, int thickness);
    
    GameGrid2D(Coord2D dimensions, int thickness, int landmarks) : Grid2D::Grid2D(dimensions) {
        // cout<<"test1"<<endl;
        init(thickness, landmarks);
        // cout<<"test2"<<endl;
    }

    GameGrid2D(const Grid2D other) : Grid2D::Grid2D(other) {
        init(3, 6);
    }

    ~GameGrid2D(){
        //Thi is a deconstructor    
    }

private: 
    const int BASE_WIDTH = 8;
    Coord2D p1UpRight;
    Coord2D p2LowLeft;
    void drawBases() {
        int gridX = getGridDimensions().getX();
        int gridY = getGridDimensions().getY();
        
        Coord2D p1LowLeft = Coord2D(0, 0);
        this->p1UpRight = Coord2D(p1LowLeft.getX() + BASE_WIDTH,
                                        p1LowLeft.getY() + BASE_WIDTH);
        Coord2D p2UpRight = Coord2D(gridX - 1, gridY - 1);
        this->p2LowLeft = Coord2D(p2UpRight.getX() - BASE_WIDTH + 1,
                                        p2UpRight.getY() - BASE_WIDTH + 1);
        
        setTypeRect(p1LowLeft, p1UpRight, Tile::TileType::TRAVERSABLE, true);
        setTypeRect(p2LowLeft, p2UpRight, Tile::TileType::TRAVERSABLE, true);
    }
    
    void init(int thickness, int numLandmarks) {
        if(this == 0) {
            cout << "THIS IS 0"<<endl;
            exit(1);
        }
        
        // Also initializes p1UpRight and p2LowLeft
        drawBases();
        
        list<Coord2D> landmarks = getDistinctRandomPoints(numLandmarks);
        landmarks.push_front(p1UpRight);
        landmarks.push_back(p2LowLeft);
        
        assert(landmarks.size() >= 2);               
        // Draw preliminary thin paths with no layers
        cout<<"init landmarks:\n"<<endl;
        for (auto lm:landmarks) {
            cout<<lm.toString()<<endl;
        }
        vector<Path> fullPath = getFullPath(landmarks, 0);
        for (Path& p : fullPath) {
            
            p.setPathType(Tile::TileType::TRAVERSABLE, false);
        }
        // Replace all empty tiles with non-traversables
       /* for (Tile t : *(this->grid)) {
            
            if (t.getType() == Tile::TileType::EMPTY) {
                
                t.setType(Tile::TileType::NON_TRAVERSABLE);
            }
        }*/
        
        // Increase thickness of traversables
        fullPath = getFullPath(landmarks, thickness);
        // cout << "init after 2getting fullpath"<<endl;
        for (Path& p : fullPath) {
            
            p.setPathType(Tile::TileType::TRAVERSABLE, true);
        }
        // cout << "returning init"<<endl;
    }
    
    
    /**
     * Returns a list of random Coord2D objects of specified size.
     * You are guaranteed that this list will have no duplicate values,
     * and will also not contain the values p1UpRight or p2LowLeft.
     * Additionally, none of these points shall fall within the bases.
     * @param amount Number of random points to generate
:set mouse=a
     * @return List of distinct Coord2D objects
     */
    list<Coord2D> getDistinctRandomPoints(int amount) {
        
        unordered_set<Coord2D, Coord2DHasher, Coord2DComparator> pointsSet;
        list<Coord2D> pointsList;
        // Use the same Random object to ensure correct output
        
        // We use a while loop instead of a for loop
        // because there's a small chance
        // that we could accidentally generate duplicate Coord2D's
        cout <<"getDRP before while" <<endl;
        Coord2D randCoord;
        while (pointsSet.size() < amount) {
            
            randCoord = getRandomNonBase();
            
            // These two will populate pointsSet later,
            // so check for duplicates now
            if (!randCoord.equals(p1UpRight) && !randCoord.equals(p2LowLeft))
                pointsSet.insert(randCoord);
        }
        for(auto point:pointsSet) { 
            pointsList.push_back(point);
        }
        cout <<"getDRP after while" <<endl;
        // As far as this function is concerned,
        // order does not matter
       // vector<Coord2D> pointsList = new vector<Coord2D>(pointsSet);
        
        return pointsList;
    }
    
    vector<Path> getFullPath(list<Coord2D> landmarks, int thickness) {
        if(this == 0) {
            cout << "THIS IS 0"<<endl;
            exit(1);
        }

        vector<Path> paths;
        vector<Coord2D> marks;

        for(auto i : landmarks){
            marks.push_back(i);
        } 
        cout <<"gFP before for"<<endl;
        for (int i = 0; i < landmarks.size() - 1; i++) {
            
            Coord2D landmark1 = marks.at(i);
            Coord2D landmark2 = marks.at(i + 1);
            cout << "gFP before Path for i:"<<i<<endl;
            if(this == 0) {
            cout << "THIS IS 0"<<endl;
            exit(1);
            }
            Path p = Path(this, landmark1, landmark2, thickness);
            cout <<"gFP after Path"<<endl;
            paths.push_back(p);
        }
        cout <<"gFP after for"<<endl;
        return paths;
    }
    
    Coord2D getRandomNonBase() {
        
        int xGridBound = this->Grid2D::getGridDimensions().getX();
        int yGridBound = this->Grid2D::getGridDimensions().getY();
        srand(time(NULL));

        int x, y;
        
        x = rand() % xGridBound;
        
        // if x is within range of base1, y needs to dodge base1
        if (x <= p1UpRight.getX()) {
            
            y = rand() % (yGridBound - p1UpRight.getY());
            y += p1UpRight.getY();
        }
        
        // else if x is within range of base2, y needs to dodge base2
        else if (p2LowLeft.getX() <= x) {
            
            y = rand() % (p2LowLeft.getY()); // exclusive
        }
        
        // Else, y doesn't need to dodge anything
        else {
            y = rand()%(yGridBound);
        }
        
        return Coord2D(x, y);
    }
    
};    

#endif //__GAMEGRID2D_HPP__

