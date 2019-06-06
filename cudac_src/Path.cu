#include "Path.h"
#include <chrono>
// #include <time.h>

using namespace std;

// make changes to setQ, then copy back to host
__global__ void PopBestPathWhileLoop(unordered_set<Tile*>)

Path::Path(Grid2D* grid) {
    this->grid=grid;
    this->joints=new list<Coord2D>();
    this->thickness=0;
}

Path::Path(Grid2D* grid, Coord2D point1, Coord2D point2, int thickness) {
    if(this == 0) {
    cout << "THIS IS 0"<<endl;
    exit(1);
    }
    this->grid=grid;
    this->joints= new list<Coord2D>();
    this->thickness=thickness;
    populateBestPath<<<dim3(1,1,1), dim3(1,1,1)>>> (point1, point2);
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
    if(!joints->size() >= 2) {
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

void populateBestPath(Coord2D src, Coord2D dest) {
    auto startTime = chrono::steady_clock::now();
    if(this == 0) {
    cout << "THIS IS 0"<<endl;
    exit(1);
    }
    Grid2D* tempGrid = new Grid2D(*(this->grid)); // FIXME: error here?
    Tile* srcTile = tempGrid->getTile(src); // usually we don't want to work with pointers to vector elements bc vector mem is reallocated
    Tile* destTile = tempGrid->getTile(dest); // on insert/delete, but since we don't modify tempGrid, this should be fine
    srcTile->setDistance(0);
    // cout<<"\nsrcTile:\n"<<srcTile->toString()<<endl;

    if(src == dest) { cout << "Attempted autopath to the same tile"<<endl;exit(1);}
    if(srcTile->getType() == Tile::TileType::NON_TRAVERSABLE) {
        cout << "Path attempted on non-traversable srcTile" << endl;
        exit(1);
    }
    if(destTile->getType() == Tile::TileType::NON_TRAVERSABLE) {
        cout << "Path attempted on non-traversable destTile" << endl;
        exit(1);
    }

    unordered_set<Tile*, TilePtrHasher, TilePtrComparator> setQ;
    // cout<< "grid:\n"<<grid->toString()<<endl;
    // cout<<"INSERTING TILE:\n"<<(*grid->getTile(Coord2D(0,0))).toString()<<endl;
    setQ.insert(grid->getTile(Coord2D(0,0)));

    // for (Tile t: tempGrid) {
    //     if(t.getType() != Tile::TileType::NON_TRAVERSABLE) {
    //         setQ.insert(t);
    //     }
    // }
    // iterate through tiles in grid
    for(unsigned int i = 0; i < tempGrid->grid->size(); ++i) {
        for(unsigned j = 0; j < tempGrid->grid->at(i).size();++j) {
            Tile* tempTile = tempGrid->grid->at(i).at(j);
            if(tempTile->getType() != Tile::TileType::NON_TRAVERSABLE) {
                // cout<<"INSERTING TILE:\n"<<tempTile->toString()<<endl;
                setQ.insert(tempTile);
            }
        }
    }
    cout << "setQ.size():" << setQ.size() << endl;

    if(setQ.find(srcTile) == setQ.end()) {
        cout << "setQ doesn't contain srcTile" << endl;
        exit(1);
    }
    if(setQ.find(destTile) == setQ.end()) {
        cout << "setQ doesn't contain destTile" << endl;
    }

    Tile* uTile = NULL;
    // FIXME: setQ's tile distances are not set
    int currentDistance = INT_MAX;

    PopBestPathWhileLoop <<< , >>> (setQ, var, );
    
    while(!setQ.empty()) {
        int runningMin = INT_MAX;

        // here the first tile's distance should be 0
        for (Tile* t : setQ) {
            if(t->getDistance() < runningMin) {
                // cout<<"TEST"<<endl;
                runningMin = t->getDistance();
                uTile = t;
            }
        }

        if(uTile == NULL) {
            cout << "Minimum distance tile uTile not properly set" << endl;
            exit(1);
        }
        if(setQ.find(uTile) == setQ.end()) {
            cout << "setQ doesn't contain uTile " << uTile->toString() << endl;
            exit(1);
        }
        setQ.erase(uTile);

        if(uTile == destTile) { 
            break;
        }

        unordered_set<Tile*, TilePtrHasher, TilePtrComparator> uNeighbors = tempGrid->getTraversableNeighbors(*uTile->getLocation());
        // int* uNeighbors = tempGrid->getTraversibleNeighbors(*(uTile->getLocation()));

        for (Tile* thisNeighbor : uNeighbors) {
            int currentDist = uTile->getDistance() + 1;
            if (currentDist < thisNeighbor->getDistance()) {
                thisNeighbor->setDistance(currentDist);
                thisNeighbor->setPreviousTile(uTile);
            }
        }
        // for (unsigned i = 0; i < 4; ++i) {
        //     currentDistance - uTile->getDistance()+1;
        //     if(currentDistance < uNeighbors[i]->getDistance()) {
        //         uNeighbors[i]->setDistance(currentDistance);
        //         uNeighbors[i]->setPreviousTile(uTile);
        //     }
        // }
    }

    if (uTile->getPreviousTile() == NULL && uTile != srcTile) {
        cout << "Condition specified by Dijkstra's not met" << endl;
        exit(1);
    }

    while(uTile != NULL) {
        joints->push_back(*uTile->getLocation());
        uTile = uTile->getPreviousTile();

        if (uTile == NULL && joints->size() < 2) {
            cout << "Not enough prev's? For sure not enough joints\nPerhaps src and dest are the same?\nsrc: " << src.toString() << "\n" <<
            "dest: " << dest.toString() << "\n" <<
            "src.equals(dest)? " << src.equals(dest);

            exit(1);
        }
    }
    // delete tempGrid
    for(unsigned i = 0; i < tempGrid->grid->size();++i) {
        for(unsigned j = 0; j < tempGrid->grid->at(i).size();++j) {
            delete tempGrid->grid->at(i).at(j);
        }
    }
    auto endTime = chrono::steady_clock::now();
    cout << "Populate Best Path took: " << chrono::duration<double, milli> (endTime-startTime).count() << endl;
}