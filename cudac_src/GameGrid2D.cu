#include "GameGrid2D.h"
#include "kernel.cu"

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
/*
*AllocateAndCall()
* PARAMETERS
*  paths -- reference to vector containing the paths that need to be filled by the GPU
*  srcs  -- array of all of our src nodes (where each path begins from)
*  dests -- array of all of our destinatio nnode (where each path ends)
*  path_sz -- how many paths we are generating. paths.size()
* FUNCTION
*  AllocateAndCall creates 4 int arrays. each index in these arrays represents a node in either
      dests or srcs. The arrays store the x coordiante and y coordinate of each node, and pass those
      along to the kernel. It also keeps track of the total amoutn of nodes across each path
*/
void GameGrid2D::AllocateAndCall(vector<Path> &paths, Coord2D* srcs, Coord2D* dests, int path_sz){
//Allocate CPU memory
    int* sizes = (int*) malloc(sizeof(int)*(path_sz-1));
    int* sizes_h = (int*) malloc(sizeof(int)*(path_sz-1));
    int* srcsX = (int*) malloc(sizeof(int)*(path_sz-1));
    int* srcsY = (int*) malloc(sizeof(int)*(path_sz-1));
    int* destsX = (int*) malloc(sizeof(int)*(path_sz-1));
    int* destsY = (int*) malloc(sizeof(int)*(path_sz-1));

    unsigned int totalSize = 0; //Tracks how large routesX and routeY should be (see below)
    for(unsigned i = 0; i < path_sz-1; i++){
	int s = determineSize(srcs[i], dests[i]); //Determines how many nodes will paths.at(i) will have
        sizes[i] = s; //For use when we copy data over the paths
        totalSize += s; //Used to initalize 
        sizes_h[i] = totalSize; //Each thread will use this to populate a segments of routes_dx and routes_dy
	srcsX[i] = srcs[i].first;  //The following will be used for some calculations in the kernel
	srcsY[i] = srcs[i].second; 
	destsX[i] = dests[i].first; 
	destsY[i] = dests[i].second; 
    }
//Allocate memory for the coordiante arrays to be used when we populate the paths
    int* routesX = (int*) malloc(sizeof(int)*totalSize);
    int* routesY = (int*) malloc(sizeof(int)*totalSize);

    int* sizes_d;
    int* srcs_dx;
    int* srcs_dy;
    int* dests_dx;
    int* dests_dy;
    int* routes_dx;
    int* routes_dy;

//numBlocks and block size
    int blockSize;
    int numBlocks;
    if(path_sz-1 > 32){blockSize = 32; numBlocks = ((path_sz-1)/blockSize)+1;}
    else{blockSize = path_sz-1; numBlocks = 1;}
 
    //Copy sizes to device
    cudaMalloc((void**) &sizes_d, sizeof(int)*(path_sz-1));
    cudaMemcpy(sizes_d, sizes_h, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    //Allocate device memory for srcs and copy to device
    cudaMalloc((void**) &srcs_dx, sizeof(int)*(path_sz-1));
    cudaMalloc((void**) &srcs_dy, sizeof(int)*(path_sz-1));
    cudaMemcpy(srcs_dx, srcsX, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    cudaMemcpy(srcs_dy, srcsY, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    //Allocate device memory for dests and copy to device
    cudaMalloc((void**) &dests_dx, sizeof(int)*(path_sz-1));
    cudaMalloc((void**) &dests_dy, sizeof(int)*(path_sz-1));
    cudaMemcpy(dests_dx, destsX, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    cudaMemcpy(dests_dy, destsY, sizeof(int)*(path_sz-1), cudaMemcpyHostToDevice);
    //Create Space for routes and copy to device
    cudaMalloc((void**) &routes_dx, sizeof(int)*totalSize);
    cudaMalloc((void**) &routes_dy, sizeof(int)*totalSize);
    cudaMemcpy(routes_dx, routesX, sizeof(int)*totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(routes_dy, routesY, sizeof(int)*totalSize, cudaMemcpyHostToDevice);

//kernel call
    getPaths<<<numBlocks, blockSize>>> (totalSize, routes_dx, routes_dy, srcs_dx, srcs_dy, dests_dx, dests_dy, sizes_d);
    cudaDeviceSynchronize(); //just to be safe

//Copy memory from device to host
    cudaMemcpy(routesX, routes_dx, sizeof(int)*totalSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(routesY, routes_dy, sizeof(int)*totalSize, cudaMemcpyDeviceToHost);

    int x_cnt = 0;
    int y_cnt = 0;
//Populate paths
    for(int i = 0; i < path_sz-1; i++){
	for(int j = 0; j <= sizes[i]; j++){
                paths.at(i).addJoint(Coord2D(routesX[x_cnt], routesY[y_cnt]));
		x_cnt++;
		y_cnt++;
        }

    }

//Free memory -- may not be necessary for CPU see the data should die by the block
    free(sizes);
    free(sizes_h);
    free(srcsX);
    free(srcsY);
    free(destsX);
    free(destsY);
    free(routesX);
    free(routesY);
    cudaFree(sizes_d);
    cudaFree(srcs_dx);
    cudaFree(srcs_dy);
    cudaFree(dests_dx);
    cudaFree(dests_dy);
    cudaFree(routes_dx);
    cudaFree(routes_dy);
}

int GameGrid2D::determineSize(Coord2D c1, Coord2D c2){
    int rows = abs(c2.second - c1.second)+1; //The +1 fixes an off-by-one error
    int cols = abs(c2.first - c1.first)+1;
    return rows+cols-1;
}



void GameGrid2D::swapSrc(Coord2D* src, Coord2D* dest){

	Coord2D* temp = src;
        src = dest;
        dest = temp;

}
