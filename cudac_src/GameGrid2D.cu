#ifndef __TEST__    
#define __TEST__

#include <utility>
#include "GameGrid2D.h"
#include "kernel.cu"
#include "support.h"

extern void bellman_ford(int,int,int,int*,int*);

void GameGrid2D::populateBestPath(Path& p) {
    // Grid2D* tempGrid = new Grid2D(p.grid); // FIXME: error here?
    // Tile* srcTile = tempGrid->getTile(p.src); // usually we don't want to work with pointers to vector elements bc vector mem is reallocated
    // Tile* destTile = tempGrid->getTile(p.dst); // on insert/delete, but since we don't modify tempGrid, this should be fine
    // srcTile->setDistance(0);
    // // cout<<"\nsrcTile:\n"<<srcTile->toString()<<endl;

    // if(p.src == p.dst) { cout << "Attempted autopath to the same tile"<<endl;exit(1);}
    // if(srcTile->getType() == Tile::TileType::NON_TRAVERSABLE) {
    //     cout << "Path attempted on non-traversable srcTile" << endl;
    //     exit(1);
    // }
    // if(destTile->getType() == Tile::TileType::NON_TRAVERSABLE) {
    //     cout << "Path attempted on non-traversable destTile" << endl;
    //     exit(1);
    // }

    // std::set<Tile*> setQ;
    // setQ.insert(p.grid->getTile(0, 0));

    // for(unsigned int i = 0; i < tempGrid->grid->size(); ++i) {
    //     for(unsigned j = 0; j < tempGrid->grid->at(i).size();++j) {
    //         //Tile* tempTile = tempGrid->grid->at(i).at(j);
    //         Tile* tempTile = tempGrid->getTile(i, j);
	// 		if(tempTile->getType() != Tile::TileType::NON_TRAVERSABLE) {
    //             setQ.insert(tempTile);
    //         }
    //     }
    // }

    // if(setQ.find(srcTile) == setQ.end()) {
    //     cout << "setQ doesn't contain srcTile" << endl;
    //     exit(1);
    // }
    // if(setQ.find(destTile) == setQ.end()) {
    //     cout << "setQ doesn't contain destTile" << endl;
    // } //Line 155

    // Tile* uTile = nullptr;

    /** Need to make tempGrid
     * tempGrid takes src and dst locations and constructs a standardized grid like this:
     * ----------dst
     * |          |
     * |          |
     * src---------
     **/
     cout << "src, dst: ";
     cout << p.src << ", " << p.dst<<endl;
     cout << "BEFORE CONSTRUCTOR" << endl;
     Grid2D tempGrid = Grid2D(p.src, p.dst);
     cout << "AFTER CONSTRUCTOR" << endl;
    //setQ holds all traversable tiles
    int threadsPerBlock;    
    int blocksPerGrid;

    int tempGridRows = tempGrid.grid->size(); 
    int tempGridCols = tempGrid.grid->at(0).size();
    int numVertices = tempGridCols*tempGridRows; // numVertices should be total num of tiles in grid; numVertices = rows*cols
    int edgesMat[numVertices][numVertices];
    int edgesArr[numVertices*numVertices]; // flattened version of edgesMatrix
    // pair<int, pair<int,int> > distances[numVertices]; // distances holds pairs containing distance from src and a pair of location coordinates for that tile
    int distances[numVertices]; // initialized in kernel

    threadsPerBlock = 256; // each thread handles an edge
    blocksPerGrid = ceil(float(numVertices)/threadsPerBlock);


    // Populate the edges matrix
    // initialize edge values using iterator
    for(int* it = &edgesMat[0][0]; it != &edgesMat[0][0] + numVertices*numVertices; ++it) {
        *it = 1000000; // initialize edge distances to 1,000,000
    }
    int i = 0; 
    int j = 0;
    for(i=0; i < numVertices; ++i) {
        for(j=0; j < numVertices; ++j) {

            // mark distances of neighbors to 1
            // neighbors are left, right, up, down
            // check bounds
            if(i == j) {
                if(i-tempGridRows >= 0) edgesMat[i-tempGridRows][j] = 1; // distances: tile -> neighbor above *in grid*
                if(i+tempGridRows < numVertices) edgesMat[i+tempGridRows][j] = 1; // distance: tile -> neighbor below in grid

                // for neighbors adjacent left/right, make sure the current tile is 
                // not on edge (tiles on left/right edges have no left/right neighbors)
                if(i-1 >= 0 && (i % tempGridCols != 0)) edgesMat[i-1][j] = 1; // distance: tile -> neighbor adjacent left in grid
                if(i+1 < numVertices && ((i+1) % tempGridCols != 0)) edgesMat[i+1][j] = 1; // distance: tile -> neighbor adjacent right in grid
                

                // the following are the exact same thing, but we still need them for bellman ford's alg
                // theres probably a way to be more efficient about this...
                if(j-tempGridRows >= 0) edgesMat[i][j-tempGridRows] = 1; 
                if(j+tempGridRows < numVertices) edgesMat[i][j+tempGridRows] = 1;
                if(j-1 >= 0 && (j % tempGridCols != 0)) edgesMat[i][j-1] = 1;
                if(j+1 < numVertices && ((j+1) % tempGridCols == 0)) edgesMat[i][j+1] = 1;


                edgesMat[i][j] = 0; // distance from tile to itself
            }
        }
    }

    // copy into edges array
    i = 0; // index for edges array
    for(int* it = &edgesMat[0][0]; it != &edgesMat[0][0]+numVertices*numVertices;++it) {
        edgesArr[i] = *it;
        ++i;
    }

    // std::cout << "Matrices:" << std::endl;
    // for(i=0; i < numVertices; ++i) {
    //     for(j=0; j < numVertices; ++j) {
    //         std::cout << edgesMat[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // exit(0);

    // initialize distances[]


    cudaError_t cuda_ret;

    cudaDeviceSynchronize();

    bellman_ford(blocksPerGrid, threadsPerBlock, numVertices, edgesArr, distances);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
        FATAL("Unable to launch kernel");
    }

    cout << "RETURN FROM KERNEL" << endl;
    cout << "tempGrid dim: " << tempGridRows << " x "<< tempGridCols <<endl;
    for (i = 0; i < numVertices; ++i) {
        if(i % tempGridCols == 0) {
            cout << endl;
        }
        if(distances[i] < 10) {
            cout << distances[i] << "   ";
        }
        else {
            cout << distances[i] << "  ";
        }
    }
    cout << endl;
    exit(0);

    // distances has shortest dist from start to other nodes
    // construct joints from distances
    // FIXME: finish
    /*BEGIN DIJKSTRAS
    
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

        set<Tile*> uNeighbors = tempGrid->getTraversableNeighbors(*uTile->getLocation());

        for (Tile* thisNeighbor : uNeighbors) {
            int currentDist = uTile->getDistance() + 1;
            if (currentDist < thisNeighbor->getDistance()) {
                thisNeighbor->setDistance(currentDist);
                thisNeighbor->setPreviousTile(uTile);
            }
        }
    }
    // END DIJKSTRAS*/

    // if (uTile->getPreviousTile() == NULL && uTile != srcTile) {
    //     cout << "Condition specified by Dijkstra's not met" << endl;
    //     exit(1);
    // }

    // while(uTile != NULL) {
    //     p.joints->push_back(*uTile->getLocation());
    //     uTile = uTile->getPreviousTile();

    //     bool arePointsSame = (p.src == p.dst);

    //     if (uTile == NULL && p.joints->size() < 2) {
    //         cerr << "Not enough prev's? For sure not enough joints\nPerhaps src and dest are the same?\nsrc: " << coord_to_string(p.src) << "\n" <<
    //         "dest: " << coord_to_string(p.dst) << "\n" <<
    //         "src.equals(dest)? " << arePointsSame;

    //         exit(1);
    //     }
    // }
    // // delete tempGrid
    // /*for(unsigned i = 0; i < tempGrid->grid->size();++i) {
    //     for(unsigned j = 0; j < tempGrid->grid->at(i).size();++j) {
    //         delete tempGrid->grid->at(i).at(j);
    //     }
    // }*/
    // delete tempGrid;
} //End populateBestPath

void GameGrid2D::drawBases() {
    int gridX = getGridDimensions().first;
    int gridY = getGridDimensions().second;
    
    Coord2D p1LowLeft = Coord2D(0, 0);
    this->p1UpRight = Coord2D(p1LowLeft.first + BASE_WIDTH,
        p1LowLeft.second + BASE_WIDTH);
    Coord2D p2UpRight = Coord2D(gridX - 1, gridY - 1);
    this->p2LowLeft = Coord2D(p2UpRight.first - BASE_WIDTH + 1,
        p2UpRight.second - BASE_WIDTH + 1);
    
    setTypeRect(p1LowLeft, p1UpRight, Tile::TileType::TRAVERSABLE, true);
    setTypeRect(p2LowLeft, p2UpRight, Tile::TileType::TRAVERSABLE, true);
}

void GameGrid2D::init(int thickness, int numLandmarks) {
    if(this == 0) {
        cout << "THIS IS 0"<<endl;
        exit(1);
    }
    
    // Also initializes p1UpRight and p2LowLeft
    drawBases();
    set<Coord2D> pointsSet;
    list<Coord2D> landmarks = getDistinctRandomPoints(numLandmarks, pointsSet);
    landmarks.push_front(p1UpRight);
    landmarks.push_back(p2LowLeft);
    
    assert(landmarks.size() >= 2);               
    // Draw preliminary thin paths with no layers
    //cout<<"init landmarks:\n"<<endl;
    //for (auto lm:landmarks) {
    //    cout<<lm.toString()<<endl;
    //}
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
    for(unsigned j = 0; j < this->grid->size(); j++){
        for(unsigned k = 0; k < this->grid->at(j).size(); k++){
            if(this->grid->at(j).at(k)->getType() == Tile::TileType::EMPTY){
             this->grid->at(j).at(k)->setType(Tile::TileType::NON_TRAVERSABLE);  
            }
        }
    }

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
 * @return List of distinct Coord2D objects
 */
std::list<Coord2D> GameGrid2D::getDistinctRandomPoints(unsigned int amount, std::set<Coord2D>&pointsSet)
{
    // unordered_set<Coord2D, Coord2DHasher, Coord2DComparator> pointsSet;
    list<Coord2D> pointsList;
        // Use the same Random object to ensure correct output

        // We use a while loop instead of a for loop
        // because there's a small chance
        // that we could accidentally generate duplicate Coord2D's
    //cout <<"getDRP before while" <<endl;
    Coord2D randCoord;
	milliseconds ms = duration_cast< milliseconds>(system_clock::now().time_since_epoch());
    srand(ms.count());
    while (pointsSet.size() < amount) {

        randCoord = getRandomNonBase();

            // These two will populate pointsSet later,
            // so check for duplicates now
        if ((randCoord != p1UpRight) && (randCoord != p2LowLeft))
            pointsSet.insert(randCoord);
    }
    for(auto point:pointsSet) { 
        pointsList.push_back(point);
    }
    //cout <<"getDRP after while" <<endl;
        // As far as this function is concerned,
        // order does not matter
       // vector<Coord2D> pointsList = new vector<Coord2D>(pointsSet);

    return pointsList;
}

vector<Path> GameGrid2D::getFullPath(list<Coord2D> landmarks, int thickness) {
    if(this == 0) {
        cout << "THIS IS 0"<<endl;
        exit(1);
    }

    vector<Path> paths;
    vector<Coord2D> marks;

    for(auto i : landmarks){
        marks.push_back(i);
    } 
    //cout <<"gFP before for"<<endl;
    unsigned int one = 1;	// done to surpress warning from -Wall compiler flag
    for (unsigned int i = 0; i < landmarks.size() - one; i++) {

        Coord2D landmark1 = marks.at(i);
        Coord2D landmark2 = marks.at(i + 1);
        //cout << "gFP before Path for i:"<<i<<endl;
        if(this == 0) {
            cout << "THIS IS 0"<<endl;
            exit(1);
        }
        Path p = Path(this, landmark1, landmark2, thickness);
        populateBestPath(p);
        //cout <<"gFP after Path"<<endl;
        paths.push_back(p);
    }
    //cout <<"gFP after for"<<endl;
    return paths;
}

Coord2D GameGrid2D::getRandomNonBase() {

    int xGridBound = this->Grid2D::getGridDimensions().first;
    int yGridBound = this->Grid2D::getGridDimensions().second;

    int x, y;

    x = rand() % xGridBound;

        // if x is within range of base1, y needs to dodge base1
    if (x <= p1UpRight.first) {

        y = rand() % (yGridBound - p1UpRight.second);
        y += p1UpRight.second;
    }

        // else if x is within range of base2, y needs to dodge base2
    else if (p2LowLeft.first <= x) {

            y = rand() % (p2LowLeft.second); // exclusive
    }
    
    // Else, y doesn't need to dodge anything
    else {
        y = rand()%(yGridBound);
    }
    
    return Coord2D(x, y);
}
#endif