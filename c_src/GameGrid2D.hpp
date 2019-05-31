#ifndef __GAMEGRID2D_H__
#define __GAMEGRID2D_H__
#include <bits/stdc++.h>
#include <stdlib.h>
#include <time.h>
#include <set>
#include <vector>
#include "Coord2D.hpp"
#include "Grid2D.hpp"
#include "Hash.hpp"
using namespace std;
using namespace Grid2d;

public class GameGrid2D {

    private const int BASE_WIDTH = 8;
    private Coord2D p1UpRight;
    private Coord2D p2LowLeft;


    private void drawBases() {
        
        int gridX = getGridDimensions().getX();
        int gridY = getGridDimensions().getY();
        
        Coord2D p1LowLeft = new Coord2D(0, 0);
        this.p1UpRight    = new Coord2D(p1LowLeft.getX() + BASE_WIDTH,
                                        p1LowLeft.getY() + BASE_WIDTH);
        Coord2D p2UpRight = new Coord2D(gridX - 1, gridY - 1);
        this.p2LowLeft    = new Coord2D(p2UpRight.getX() - BASE_WIDTH + 1,
                                        p2UpRight.getY() - BASE_WIDTH + 1);
        
        setTypeRect(p1LowLeft, p1UpRight, Tile.TileType.TRAVERSABLE, true);
        setTypeRect(p2LowLeft, p2UpRight, Tile.TileType.TRAVERSABLE, true);
    }
    
    private void init(int thickness, int numLandmarks) {
        
        // Also initializes p1UpRight and p2LowLeft
        drawBases();
        
        vector<Coord2D> landmarks = getDistinctRandomPoints(numLandmarks);
        landmarks.insert(0, p1UpRight);
        landmarks.insert(landmarks.size(), p2LowLeft);
        
        assert(landmarks.size() >= 2);               
        // Draw preliminary thin paths with no layers
        vector<Path> fullPath = getFullPath(landmarks, 0);
        for (Path p : fullPath) {
            
            p.setPathType(Tile.TileType.TRAVERSABLE, false);
        }
        
        // Replace all empty tiles with non-traversables
        for (Tile t : this) {
            
            if (t.getType() == Tile.TileType.EMPTY) {
                
                t.setType(Tile.TileType.NON_TRAVERSABLE);
            }
        }
        
        // Increase thickness of traversables
        fullPath = getFullPath(landmarks, thickness);
        for (Path p : fullPath) {
            
            p.setPathType(Tile.TileType.TRAVERSABLE, true);
        }
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
    private vector<Coord2D> getDistinctRandomPoints(int amount) {
        
        unordered_set<Coord2D, TileHasher, TileComparator> pointsSet;
        
        // Use the same Random object to ensure correct output
        
        // We use a while loop instead of a for loop
        // because there's a small chance
        // that we could accidentally generate duplicate Coord2D's
        while (pointsSet.size() < amount) {
            
            Coord2D randCoord = getRandomNonBase();
            
            // These two will populate pointsSet later,
            // so check for duplicates now
            if (!randCoord.equals(p1UpRight) && !randCoord.equals(p2LowLeft))
                pointsSet.insert(randCoord);
        }
        
        // As far as this function is concerned,
        // order does not matter
        vector<Coord2D> pointsList = new vector<Coord2D>(pointsSet);
        
        return pointsList;
    }
    
    private vector<Path> getFullPath(vector<Coord2D> landmarks, int thickness) {
        
        vector<Path> paths = new vector<Path>(landmarks.size());
        
        for (int i = 0; i < landmarks.size() - 1; i++) {
            
            Coord2D landmark1 = landmarks.get(i);
            Coord2D landmark2 = landmarks.get(i + 1);
            
            Path p = new Path(this, landmark1, landmark2, thickness);
            paths.add(p);
        }
        
        return paths;
    }
    
    private Coord2D getRandomNonBase() {
        
        int xGridBound = getGridDimensions().getX();
        int yGridBound = getGridDimensions().getY();
        srand(time(null));

        int x, y;
        
        x = rand() % xGridBounds;
        
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
        
        return new Coord2D(x, y);
    }
    
    public GameGrid2D(Coord2D dimensions, int thickness, int landmarks) : Grid2D(dimensions) {
        init(thickness, landmarks);
    }

    public GameGrid2D(Grid2D other) : Grid2D(other) {
        init(3, 6);
    }

    public ~GameGrid2d(){
        //This is a deconstructor
        
    }
}    
#endif    

