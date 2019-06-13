#include "GameGrid2D.h"
#include "kernel.cu"

void GameGrid2D::populateBestPath(Path& p) {
    Grid2D* tempGrid = new Grid2D(p.src, p.dst);
    int INF = 1000000;
    unsigned int numVerts = tempGrid->size();
    int *distances;
    int *edges;
    Tile* srcTile;
    Tile* destTile;
    //Initialize source and desitination Tile
    if(tempGrid->checkIsMirrored()){
        srcTile = tempGrid->getTile(p.dst); 
        destTile = tempGrid->getTile(p.src);
    }
    else{
        srcTile = tempGrid->getTile(p.src); 
        destTile = tempGrid->getTile(p.dst);
    } 
    
    if(p.src == p.dst) { cout << "Attempted autopath to the same tile"<<endl;exit(1);}
    if(srcTile->getType() == Tile::TileType::NON_TRAVERSABLE) {
        cout << "Path attempted on non-traversable srcTile" << endl;
        exit(1);
    }
    if(destTile->getType() == Tile::TileType::NON_TRAVERSABLE) {
        cout << "Path attempted on non-traversable destTile" << endl;
        exit(1);
    }

    distances = (int*) malloc(sizeof(unsigned int)*numVerts);
    edges = (int*) malloc(sizeof(int)*numVerts*numVerts);
    int fromX = 0;
    int fromY = 0;
    int toX = 0;
    int toY = 0;
    //Initialize all edges with either 1 if the two vertices are adjacent
    // or INF if there is no edge connecting them.
    // Vertices have no edge to themselves
    for(unsigned i = 0; i < numVerts; i++){
        for(unsigned j = 0; j < numVerts; j++){
           Coord2D* currVert = tempGrid->getTile(fromY, fromX)->getLocation();
           Coord2D* otherVert = tempGrid->getTile(toY, toX)->getLocation();
           if(areNeighbors(*currVert, *otherVert)){
               edges[i*numVerts+j] = 1;
           }
           else{
               edges[i*numVerts+j] = INF;
           } 
           
           if(j >= tempGrid->getROWS())
               fromY = j/tempGrid->getROWS();
           toX = j%tempGrid->getCOLS();
        }

        if(i >= tempGrid->getROWS())
            fromY = i/tempGrid->getROWS();
        fromX = i%tempGrid->getCOLS();
        toX = 0;
        toY = 0; 
    } //End of edge initialization
    int threadsPerBlock = 256; //Standard amount of threads per block
    int blocksPerGrid = (numVerts/threadsPerBlock) + 1;
   
/*
 * Error checking
 */
    bellman_ford(blocksPerGrid, threadsPerBlock, numVerts, edges, distances);
//End check
//
/* THIS IS LEGACY FOR USING DIJKTRA'S
    std::set<Tile*> setQ;
    setQ.insert(p.grid->getTile(0, 0));

    for(unsigned int i = 0; i < tempGrid->grid->size(); ++i) {
        for(unsigned j = 0; j < tempGrid->grid->at(i).size();++j) {
            //Tile* tempTile = tempGrid->grid->at(i).at(j);
            Tile* tempTile = tempGrid->getTile(i, j);
			if(tempTile->getType() != Tile::TileType::NON_TRAVERSABLE) {
                setQ.insert(tempTile);
            }
        }
    }
*/
/*
 * Error Checking
 */
/* LEGACY CODE
    if(setQ.find(srcTile) == setQ.end()) {
        cout << "setQ doesn't contain srcTile" << endl;
        exit(1);
    }
    if(setQ.find(destTile) == setQ.end()) {
        cout << "setQ doesn't contain destTile" << endl;
    } //Line 155
*/
//End Check 



 //   Tile* uTile = nullptr;
/*
THIS IS ALL MARCOS' CODE. I have preserved it just in case.
    //setQ holds all traversable tiles
    int threadsPerBlock;    
    int blocksPerGrid;
    int numVertices = setQ.size();
    int edgesMat[numVertices][numVertices];
    int edgesArr[numVertices*numVertices];
    int distances[numVertices];

    threadsPerBlock = 256; // each thread handles an edge
    blocksPerGrid = ceil(float(n)/threadsPerBlock);

    // Populate the edges matrix
    // initialize edge values using iterator
    for(int* it = &edgesArr[0][0]; it != &edgesArr[0][0] + numVertices*numVertices; ++it) {
        *it = 1000000; // initialize edge distances to 1,000,000
    }
    int i = 0; 
    int j = 0;
    for (Tile* t: tempGrid->grid) {
        // mark distances of neighbors to 1
        set<Tile*> neighbors = t->getTraversableNeighbors();
        for(auto n : neighbors) {
            edges[i][j++] = 1; // i is index of current tile, j is neighbor
        }
        ++i;
    }
    // copy into edges array
    i = 0; // index for edges array
    for(int* it = &edges[0][0]; it != &edges[0][0]+numVertices*numVertices;++it) {
        edges[i] = *it;
        ++i;
    }
*/ 

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
    /*
    Coord2D* coordGrid[this->getROWS()*this->getCOLS()];
    for(unsigned i = 0; this->getROWS(); i++){
        for(unsigned j = 0; j < this->getCOLS(); j++){
            coordGrid[i*this->getROWS()+j]  = Coord2D(j, i);
        }
    } */
    vector<Path> paths;
    vector<Coord2D> marks;
    
    for(auto i : landmarks){
        marks.push_back(i);
    } 
    vector<Path> routes;
    Coord2D srcs[landmarks.size()-1];
    Coord2D dests[landmarks.size()-1];
    //cout <<"gFP before for"<<endl;
    unsigned int one = 1;	// done to surpress warning from -Wall compiler flag
    for (unsigned int i = 0; i < marks.size() - one; i++) {
        srcs[i].first = marks.at(i).first;
        srcs[i].second = marks.at(i).second;
        dests[i].first =  marks.at(i + 1).first;
        dests[i].second =  marks.at(i + 1).second;
        routes.push_back(Path(this,srcs[i], dests[i], thickness));
    }
//Break point 0
    AllocateAndCall(routes,(Coord2D*) &srcs,(Coord2D*) &dests, marks.size());
    for(unsigned int k = 0; k < marks.size()-1;k++){
	paths.push_back(routes.at(k)); //This currently segfaults
    }
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

void GameGrid2D::AllocateAndCall(vector<Path> &paths, Coord2D* srcs, Coord2D* dests, int path_sz){
//Breakpoint 1
    int* sizes = (int*) malloc(sizeof(int)*(path_sz-1));
    int* sizes_h = (int*) malloc(sizeof(int)*(path_sz-1));
    int* srcsX = (int*) malloc(sizeof(int)*(path_sz-1));
    int* srcsY = (int*) malloc(sizeof(int)*(path_sz-1));
    int* destsX = (int*) malloc(sizeof(int)*(path_sz-1));
    int* destsY = (int*) malloc(sizeof(int)*(path_sz-1));
    unsigned int totalSize = 0;
    for(unsigned i = 0; i < path_sz-1; i++){
	int s = determineSize(srcs[i], dests[i]);
        sizes[i] = s;
        totalSize += s;
        sizes_h[i] = totalSize;
	srcsX[i] = srcs[i].first; 
	srcsY[i] = srcs[i].second; 
	destsX[i] = dests[i].first; 
	destsY[i] = dests[i].second; 
    }

    int* routesX = (int*) malloc(sizeof(int)*totalSize);
    int* routesY = (int*) malloc(sizeof(int)*totalSize);

    int* sizes_d;
    int* srcs_dx;
    int* srcs_dy;
    int* dests_dx;
    int* dests_dy;
    int* routes_dx;
    int* routes_dy;
    //Copy sizes to device
    cudaMalloc((void**) &sizes_d, sizeof(int)*(path_sz-1));
    cudaMemcpy(sizes_d, sizes_h, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    //Copy srcs to device
    cudaMalloc((void**) &srcs_dx, sizeof(int)*(path_sz-1));
    cudaMalloc((void**) &srcs_dy, sizeof(int)*(path_sz-1));
    cudaMemcpy(srcs_dx, srcsX, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    cudaMemcpy(srcs_dy, srcsY, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    //Copy dests to device
    cudaMalloc((void**) &dests_dx, sizeof(int)*(path_sz-1));
    cudaMalloc((void**) &dests_dy, sizeof(int)*(path_sz-1));
    cudaMemcpy(dests_dx, destsX, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    cudaMemcpy(dests_dy, destsY, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    //Create Space for routes
    cudaMalloc((void**) &routes_dx, sizeof(int)*totalSize);
    cudaMalloc((void**) &routes_dy, sizeof(int)*totalSize);
    cudaMemcpy(routes_dx, routesX, sizeof(int)*totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(routes_dy, routesY, sizeof(int)*totalSize, cudaMemcpyHostToDevice);
//BreakPoint 2
    getPaths<<<1, 8>>> (totalSize, routes_dx, routes_dy, srcs_dx, srcs_dy, dests_dx, dests_dy, sizes_d);
    cudaDeviceSynchronize();
    cudaMemcpy(routesX, routes_dx, sizeof(int)*totalSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(routesY, routes_dy, sizeof(int)*totalSize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < totalSize; i++){

        printf("routesX[%d]: %d   ", i, routesX[i]);
        printf("routesY[%d]: %d\n", i, routesY[i]);
    } 
    int x_cnt = 0;
    int y_cnt = 0;
    for(int i = 0; i < path_sz-1; i++){
	for(int j = 0; j <= sizes[i]; j++){
                printf("In path allocation loop level %d\n", x_cnt);
		paths.at(i).joints->push_back(Coord2D(routesX[x_cnt], routesY[y_cnt]));
		x_cnt++;
		y_cnt++;
        }

    }
    free(sizes);
    cudaFree(sizes_d);
}

int GameGrid2D::determineSize(Coord2D c1, Coord2D c2){
    int rows = abs(c2.second - c1.second)+1;
    int cols = abs(c2.first - c1.first)+1;
    return rows+cols-1;
}



void GameGrid2D::swapSrc(Coord2D* src, Coord2D* dest){

	Coord2D* temp = src;
        src = dest;
        dest = temp;

}
