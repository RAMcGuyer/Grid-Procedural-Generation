#include <iostream>
#include <unordered_set>
#include <set>
#include <chrono>
#include <thread>
#include <cassert>

#include "Coord2D.hpp"
#include "Tile.hpp"
#include "Path.hpp"
#include "Grid2D.hpp"

using namespace std;

void testGenerateGameGrid(int numOfGrids);
void testDijkstra();
void testIterator();
void testPaths();
void testMarkRect();
void testMarkRow();
void testGrid();
void testSetOfPoints();

int main(int argc, char* argv) {
    cout << "Hello World!" << endl;
    if(arc > 1) {
        cout << "Too many arguments!" << endl;
        exit(1);
    }

    if(argc <= 1) {
        testGenerateGameGrid(INT_MAX);
    }
    else {
        int numOfGrids = argv[1];
        testGenerateGameGrid(numOfGrids);
    }
}

void testGenerateGameGrid(int numOfGrids) {
    if(numOfGrids < 0) {
        cout << "You tried to generate a negative number of grids" << endl;
        exit(1);
    }

    Coord2D gridDimensions = Coord2D(50,50);
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
}

void testDijkstra() {
    Coord2D gridDimensions = Coord2D(10,10);
    Grid2D grid = Grid2D(gridDimensions);

    Coord2D obstacle_LowerLeft = Coord2D(5,2);
    Coord2D obstacle_UpperRight = Coord2D(7,5);

    grid.setTypeRect(obstacle_LowerLeft, obstacle_UpperRight, TileType::NON_TRAVERSABLE, true);
    cout << "Grid with single obstacle:\n\n" << grid.toString() << endl;

    Coord2D pointA = Coord2D(1,1);
    Coord2D pointB = Coord2D(9,8);
    grid.getTile(pointA).setType(TileType::TRAVERSABLE);
    grid.getTile(pointB).setType(TileType::TRAVERSABLE);

    cout << "Grid with pointA = " << pointA.toString() << endl;
    cout << "          pointB = " << pointB.toString() << endl;
    cout << grid.toString() << endl;

    Path path = Path(grid, pointA, pointB, 1);
    path.setPathType(TileType::TRAVERSABLE, false);

    cout << "Grid with best route:\n" << grid.toString() << endl;
}

void testIterator() {
    Coord2D gridDimensions = new Coord2D(50,50);
    Grid2D grid = new Grid2D(gridDimensions);

    unordered_set<Tile> tiles;
    tiles.insert(grid.getTile(new Coord2D(0,0)));
    for (Tile t : grid) {
        tiles.insert(t);
    }
    cout << "Expected size: " << gridDimensions.getX()*gridDimensions.getY() << endl;
    cout << "Num of tiles: " << tiles.size() << endl;

    delete gridDimensions;
    delete grid;
}

void testPaths() {
    Grid2D grid = Grid2D(Coord2D(50,50));
    cout << "Empty grid: \n\n" << grid.toString() << endl;

    Path path = Path(grid);
    assert(path.addJoint(Coord2D(10,1));
    assert(path.addJoint(Coord2D(10, 2)));
    assert(path.addJoint(Coord2D(30, 2));
    assert(path.addJoint(Coord2D(30, 0)));

    path.setPathType(TileType::TRAVERSABLE, true);

    Grid2D copy = Grid2D(grid);
    copy.setTile(TileType::TRAVERSABLE, Coord2D(0,0));

    cout << "Populated grid:\n\n" << grid.toString() << endl;
    cout << "Copy:\n\n" << copy.toString() << endl;
}

void testMarkRect() {
    Coord2D gridDimensions = Coord2D(7,13);
    Grid2D grid = Grid2D(gridDimensions);

    cout << "Empty grid:\n" << grid.toString() << endl;

    Coord2D middlBand_lowLeft = Coord2D(0,3);
    Coord2D middleBand_upRight = Coord2D(gridDimensions.getX()-1, 6);

    cout << "Marking a band in the middle:") <<endl;
    grid.markRect(middleBand_lowLeft, middleBand_upRight, true);

    cout << grid.toString() << endl;
}

void testMarkRow() {
    Coord2D gridDimensions = Coord2D(10,10);
    Grid2D = Grid2D(gridDimensions);

    cout << "Empty grid:\n" << grid.toString() <<endl;

    Coord2D lowbar_left = Coord2D(0,2);
    Coord2D lowbar_right = Coord2D(gridDimensions.getX()-1, lowbar_left.getY());

    cout << "Marking lower bar..." <<endl;
    grid.setTypeLine(lowbar_left, lowbar_right, TileType::TRAVERSAGBLE, 0,true);
    cout << grid.toString() <<endl;

    Coord2D vertbar_down = Coord2D(2,0);
    Coord2D vertbar_up = Coord2D(verbar_down.getX(), gridDimensions.getY()-1);

    cout << "Marking vert bar..." << endl;
    grid.setTypeLine(vertbar_down, vertbar_up, TileType::TRAVERSABLE,2,true);
    cout << grid.toString() <<endl;
}

void testGrid() {
    Coord2D gridDimensions = Coord2D(7,13);
    Coord2D testPoint = Coord2D(0,gridDimensions.getY()-1);
    Coord2D emptyPoint = Coord2D(3,10);
    Coord2D failPoint = Coord2D(7,14);

    Grid2D grid = Grid2D(gridDimensions);
    grid.setTile(TileType::TRAVERSABLE, testPoint);

    cout << grid.toString() << endl;
    cout << "Tile at " << testPoint.toString() << ": " << grid.getTile(testPoint) << endl;
    cout << "Tile at " << emptyPoint.toString() << ": " << grid.getTile(emptyPoint) << endl;

    cout << "Can testPoint go up? " << (grid.canGoUp(testPoint) ? "true" : "false") << endl;
    cout << "Can testPoint go down? " << (grid.canGoDown(testPoint) ? "true" : "false") << endl;
    cout << "Can testPoint go left? " << (grid.canGoLeft(testPoint) ? "true" : "false") << endl;
    cout << "Can testPoint go right? " << (grid.canGoRight(testPoint) ? "true" : "false") << endl;

    cout << "Getting the up of testPoint:" << grid.getUp(testPoint).toString() << endl;
    cout << "Getting testPoint manually:" << grid.getTile(testPoint).toString() <<endl;

    cout << grid.toString() << endl;
}

void testSetOfPoints() {
    Coord2D original = Coord2D(3,4);
    Coord2D duplicate = Coord2D(original);
    Coord2D anotherone = Coord2D(6,9);

    assert(&original != &duplicate);
    assert(duplicate.equals(original));

    unordered_set<Coord2D> set = unordered_set<Coord2D>(3);
    set.insert(original);
    cout << "Set contains original. Contains duplicate? " << ((set.find(duplicate) == set.end()) ? "true":"false") << endl;
    set.insert(duplicate);
    set.insert(anotherone);

    list<Coord2D> list = list<Coord2D>();
    for (Coord2D element : set) {
        list.push_back(element);
    }
    cout << "Set: ";
    for (Coord2D element : set) {
        cout << element << " ";
    }
    cout << endl;
    cout << "List: ";
    for (Coord2D element : list) {
        cout << element << " ";
    }
    cout << endl;
}