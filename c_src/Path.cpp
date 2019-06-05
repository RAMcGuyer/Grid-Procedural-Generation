#include "Path.h"

using namespace std;

Path::Path(Grid2D* grid) {
    this->grid=grid;
    this->joints=new list<Coord2D>();
    this->thickness=0;
}

Path::Path(Grid2D* grid, int thickness) {
    if(this == 0) {
    cout << "THIS IS 0"<<endl;
    exit(1);
    }
    this->grid=grid;
    this->joints= new list<Coord2D>();
    this->thickness=thickness;

}

Path::Path(Grid2D* grid, list<Coord2D> & joints, int thickness) { // FIXME: how are joints passed? make new in constructor?
    this->grid=grid;
    this->joints = new list<Coord2D>(joints);
    this->thickness=thickness;
}

Path::~Path() {
    // anything here?
    // for sure dont delete joints
}

bool Path::areCompatibleJoints(Coord2D joint1, Coord2D joint2) {
    return joint1.getX() == joint2.getX() || joint1.getY() == joint2.getY();
}

bool Path::addJoint(Coord2D newJoint) {
    if (joints->empty()) {
        joints->push_back(newJoint);
        return true;
    }
    else {
        return addJoint(newJoint, joints->size());
    }
}

bool Path::addJoint(Coord2D newJoint, int index) {
    // range check index
    // passing this check means index is: 0 <= index <= joints->size()
    if(index < 0 || (unsigned)index > joints->size()) {
        return false;
    }

    // add newJoint at joints[index]
    auto it = joints->begin();
    advance(it,index); // it can be joints->begin() <= it <= joints->end()
    joints->insert(it, newJoint); // FIXME: do we need to reset "it" to joints[index]?

    // make sure joints[index] is compatible with neighbors prev/next (if they exist)
    auto prevIt = prev(it);

    // check that joint[index] has previous
    // smallest position "it" can be is joints->begin()
    if(it != joints->begin()) { 
        // if we made it here, we know joints[index] is not joints->begin(),
        // so it has a previous neighbor -> now we check compatibility
        Coord2D left_neighbor = *(prevIt);
        if(!areCompatibleJoints(left_neighbor, newJoint)) {
            it = joints->erase(it);
            return false;
        }
    }
    //reset "it" to joints[index]
    it=joints->begin();
    advance(it,index);

    // check that joint[index] has next
    // largest position "it" can be is joints->end()
    // "it" has next neighbor if: it != joints->end() and next(it) != joints->end()
    auto nextIt = next(it);
    if(it != joints->end() && nextIt != joints->end()) {
        // if we made it here, we know joints[index] has next neighbor 
        // so we can check compatibility
        Coord2D right_neighbor = *(nextIt);
        if(!areCompatibleJoints(newJoint, right_neighbor)) {
            it = joints->erase(it);
            return false;
        }
    }
    return true;
}

void Path::setPathType(Tile::TileType type, bool prioritize) {
    if(!(joints->size() >= 2)) {
        cout << "Not enough joints in path" <<endl;
        exit(1);
    }
    Coord2D firstJoint = joints->front();
    list<Coord2D>::iterator it = joints->begin();
    for(;it != joints->end();++it) {
        Coord2D secondJoint = *it;
        grid->setTypeLine(firstJoint, secondJoint, type, thickness, prioritize);
        firstJoint = secondJoint; // FIXME: possible error - does this do the same thing as in java?
    }
}

