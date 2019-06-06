#include <iostream>
#include <unordered_set>
#include <set>
#include <chrono>
#include <thread>
#include <cassert>

using namespace std;

#include "Coord2D.h"
#include "Tile.h"
#include "Path.h"
#include "Grid2D.h"
#include "GameGrid2D.h"
#include "Hash.h"

void testGenerateGameGrid(int numOfGrids) {
    if(numOfGrids < 0) {
        cout << "You tried to generate a negative number of grids" << endl;
        exit(1);
    }

    Coord2D* gridDimensions = new Coord2D(50,50);
    for(int i = 0; i < numOfGrids; ++i) {
        try
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(19));
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        GameGrid2D grid = GameGrid2D(gridDimensions, 5, 7);
        cout << "Grid #" << i << "\n" << grid.toString() << "\n" << endl;
    }
	delete gridDimensions;
}


void testIterator() {
    Coord2D* gridDimensions = new Coord2D(50,50);
    Grid2D grid2D = Grid2D(gridDimensions);

    unordered_set<Tile,TileHasher,TileComparator> tiles;
	Coord2D* emptyCoord = new Coord2D(0,0);
    tiles.insert(*grid2D.getTile(emptyCoord));
    // for (Tile t : grid) {
    //     tiles.insert(t);
    // }

    for(unsigned int i = 0; i < grid2D.grid->size(); ++i) {
        for(unsigned j = 0; j < grid2D.grid->at(i).size();++j) {
            Tile* tempTile = grid2D.grid->at(i).at(j);
            tiles.insert(*tempTile);
        }
    }

    cout << "Expected size: " << gridDimensions->getX()*gridDimensions->getY() << endl;
    cout << "Num of tiles: " << tiles.size() << endl;
	
	delete gridDimensions;
	delete emptyCoord;
}

void testPaths() {
    Coord2D* newCoord = new Coord2D(50, 50);
    Grid2D grid = Grid2D(newCoord);
    cout << "Empty grid: \n\n" << grid.toString() << endl;

    Path path = Path(&grid);
    //assert(path.addJoint(Coord2D(10,1)));
    //assert(path.addJoint(Coord2D(10, 2)));
    //assert(path.addJoint(Coord2D(30, 2)));
    //assert(path.addJoint(Coord2D(30, 0)));

    path.setPathType(Tile::TileType::TRAVERSABLE, true);

    Grid2D copy = Grid2D(grid);
	Coord2D* emptyCoord = new Coord2D(0,0);
    copy.setTile(Tile::TileType::TRAVERSABLE, emptyCoord);

    cout << "Populated grid:\n\n" << grid.toString() << endl;
    cout << "Copy:\n\n" << copy.toString() << endl;

	delete emptyCoord;
    delete newCoord;
}

void testMarkRect() {
    Coord2D* gridDimensions = new Coord2D(7,13);
    Grid2D grid = Grid2D(gridDimensions);

    cout << "Empty grid:\n" << grid.toString() << endl;

    Coord2D* middleBand_lowLeft = new Coord2D(0,3);
    Coord2D*  middleBand_upRight = new Coord2D(gridDimensions->getX()-1, 6);

    cout << "Marking a band in the middle:" <<endl;
    grid.markRect(middleBand_lowLeft, middleBand_upRight, true);

    cout << grid.toString() << endl;

    delete gridDimensions;
    delete middleBand_lowLeft;
    delete middleBand_upRight;
}

void testMarkRow() {
    Coord2D* gridDimensions = new Coord2D(10,10);
    Grid2D grid = Grid2D(gridDimensions);

    cout << "Empty grid:\n" << grid.toString() <<endl;

    Coord2D* lowbar_left = new Coord2D(0,2);
    Coord2D* lowbar_right = new Coord2D(gridDimensions->getX()-1, lowbar_left->getY());

    cout << "Marking lower bar..." <<endl;
    grid.setTypeLine(lowbar_left, lowbar_right, Tile::TileType::TRAVERSABLE, 0,true);
    cout << grid.toString() <<endl;

    Coord2D* vertbar_down = new Coord2D(2,0);
    Coord2D* vertbar_up = new Coord2D(vertbar_down->getX(), gridDimensions->getY()-1);

    cout << "Marking vert bar..." << endl;
    grid.setTypeLine(vertbar_down, vertbar_up, Tile::TileType::TRAVERSABLE,2,true);
    cout << grid.toString() <<endl;

    delete gridDimensions;
	delete lowbar_left;
	delete lowbar_right;
	delete vertbar_down;
	delete vertbar_up;
}

void testGrid() {
    Coord2D* gridDimensions = new Coord2D(7,13);
    Coord2D* testPoint = new Coord2D(0,gridDimensions->getY()-1);
    Coord2D* emptyPoint = new Coord2D(3,10);
    Coord2D* failPoint = new Coord2D(7,14);

    Grid2D grid = Grid2D(gridDimensions);
    grid.setTile(Tile::TileType::TRAVERSABLE, testPoint);

    cout << grid.toString() << endl;
    cout << "Tile at " << testPoint->toString() << ": " << grid.getTile(testPoint) << endl;
    cout << "Tile at " << emptyPoint->toString() << ": " << grid.getTile(emptyPoint) << endl;

    cout << "Can testPoint go up? " << (grid.canGoUp(testPoint) ? "true" : "false") << endl;
    cout << "Can testPoint go down? " << (grid.canGoDown(testPoint) ? "true" : "false") << endl;
    cout << "Can testPoint go left? " << (grid.canGoLeft(testPoint) ? "true" : "false") << endl;
    cout << "Can testPoint go right? " << (grid.canGoRight(testPoint) ? "true" : "false") << endl;

    cout << "Getting the up of testPoint:" << grid.getUp(testPoint)->toString() << endl;
    cout << "Getting testPoint manually:" << grid.getTile(testPoint)->toString() <<endl;

    cout << grid.toString() << endl;
	
	delete gridDimensions;
	delete testPoint;
	delete emptyPoint;
	delete failPoint;
}

void testSetOfPoints() {
    Coord2D* original = new Coord2D(3,4);
    Coord2D* duplicate = new Coord2D(original);
    Coord2D* anotherone = new Coord2D(6,9);

    assert(&original != &duplicate);
    assert(duplicate->equals(original));

    // unordered_set<Coord2D,Coord2DHasher,Coord2DComparator> set = unordered_set<Coord2D>(3);
    unordered_set<Coord2D*,Coord2DHasher,Coord2DComparator> set;
    set.insert(original);
    cout << "Set contains original. Contains duplicate? " << ((set.find(duplicate) == set.end()) ? "true":"false") << endl;
    set.insert(duplicate);
    set.insert(anotherone);

    list<Coord2D> list;
    for (Coord2D element : set) {
        list.push_back(element);
    }
    cout << "Set: ";
    for (Coord2D element : set) {
        cout << element.toString() << " ";
    }
    cout << endl;
    cout << "List: ";
    for (Coord2D element : list) {
        cout << element.toString() << " ";
    }
    cout << endl;

    delete original;
    delete duplicate;
    delete anotherone;
}

int main(int argc, char** argv) {
    cout << "Hello World!" << endl;
    if(argc > 2) {
        cout << "Too many arguments!" << endl;
        exit(1);
    }
    else if(argc == 1) {
        testGenerateGameGrid(INT_MAX);
    }
    else {
        int numOfGrids = atoi(argv[1]);
        auto startTime = std::chrono::steady_clock::now();
        testGenerateGameGrid(numOfGrids);
        auto endTime = std::chrono::steady_clock::now();
        cout << "Took " << std::chrono::duration<double,milli>(endTime-startTime).count() << "ms to generate grid(s)"<<endl;
    }
}
