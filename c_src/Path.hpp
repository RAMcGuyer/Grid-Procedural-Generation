#include <iostream>
#include <list>
#include <iterator>
#include <unordered_set>
#include "Grid2D.h"
#include "Tile.h"
using namespace std;

class Path public {
    private:
        Grid2D grid;
        list<Coord2D> joints;
        int thickness;

    public:
        Path(Grid2D grid) {
            this->grid=grid;
            joints=new list<Coord2D>();
            this->thickness=0;
        }
        Path(Grid2D grid, Coord2D point1, Coord2D point2, int thickness) {
            this->grid=grid;
            this->joints= new list<Coord2D>();
            this->thickness=thickness;
            populateBestPath(point1, point2);
        }
        Path(Grid2D grid, list<Coord2D> joints, int thickness) { // FIXME: how are joints passed? make new in constructor?
            this->grid=grid;
            this->joints = new list<Coord2D>(joints);
            this->thickness=thickness;
        }

        bool areCompatibleJoints(Coord2D joint1, Coord2D joint2) {
            return joint1.getX() == joint2.getX() || joint1.getY() == joint2.getY();
        }

        bool addJoint(Coord2D newJoint) {
            if (joints.isEmpty()) {
                joints.push_back(newJoint);
                return true;
            }
            else {
                return addJoint(newJoint, joints.size());
            }
        }

        bool addJoint(Coord2D newJoint, int index) {
            joints.insert(index, newjoin);
            list<Coord2D>::iterator it = joints.begin();
            advance(it, index); 
            auto prevIt = prev(it);
            if(prevIt >= joints.begin()) { // it has valid previous iterator
                Coord2D left_neighbor = *(prevIt);
                if(!areCompatibleJoints(left_neighbor, newJoint)) {
                    it = joints.erase(it);
                    return false;
                }
            }
            auto nextIt = next(it);
            if(nextIt < joints.end()) {// nextIt is valid
                Coord2D right_neighbor = *(nextIt);
                if(!areCompatibleJoints(newJoint, right_neighbor)) {
                    it = joints.erase(it);
                    return false;
                }
            }
            return true;
        }

        void setPathType(TileType type, bool prioritize) {
            if(!joints.size() >= 2) {
                cout << "Not enough joints in path" <<endl;
                exit(1);
            }
            Coord2D firstJoint = joints.front();
            list<Coord2D> it = joints.begin();
            for(;it != joints.end();++it) {
                Coord2D secondJoint = *it;
                grid.setTypeLine(firsJoint, secondJoint, type, thickness, prioritize);
                firstJoint = secondJoint; // FIXME: possible error - does this do the same thing as in java?
            }
        }

        void populateBestPath(Coord2D src, Coord2D dest) {
            Grid2D tempGrid = new Grid2D(this->grid);
            Tile srcTile = tempGrid.getTile(src);
            Tile destTile = tempGrid.getTile(dest);
            srcTile.setDistance(0);

            if(src == dest) { cout << "Attempted autopath to the same tile"<<endl;exit(1);}
            if(srcTile.getType() == NON_TRAVERSABLE) {
                cout << "Path attempted on non-traversable srcTile" << endl;
                exit(1);
            }
            if(destTile.getType() == NON_TRAVERSABLE) {
                cout << "Path attempted on non-traversable destTile" << endl;
                exit(1);
            }

            unordered_set<Tile> setQ = new unordered_set<Tile>();
            setQ.insert(grid.getTile(new Coord2D(0,0)));
            for (Tile t: tempGrid) {
                if(t.getType() != NON_TRAVERSABLE) {
                    setQ.insert(t);
                }
            }

            if(setQ.find(srcTile) == setQ.end()) {
                cout << "setQ doesn't contain srcTile" << endl;
                exit(1);
            }
            if(setQ.find(destTile) == setQ.end()) {
                cout << "setQ doesn't contain destTile" << endl;
            }

            Tile uTile = NULL;
        }


};