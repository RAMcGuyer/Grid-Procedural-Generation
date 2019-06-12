#include "Grid2D.h"

using namespace std;

Grid2D::Grid2D() {
	// empty default constructor
}

Grid2D::Grid2D(Coord2D dimensions) {
	assert(dimensions.first >= 1);
	assert(dimensions.second >= 1);

	ROWS = dimensions.second;
	COLS = dimensions.first;
	isMirrored = false;
	// initialize grid to 2D vector with ROWS rows and COLS columns, initialize rows to 0s
	grid = new vector<vector<Tile*> > (ROWS, vector<Tile*>(COLS,NULL));
	for (int thisRow = 0; thisRow < ROWS; thisRow++) {
		for (int thisCol = 0; thisCol < COLS; thisCol++) {
				setTile(Tile::TileType::EMPTY, Coord2D(thisCol, thisRow));
		}
	}
}

Grid2D::Grid2D(Coord2D corner1, Coord2D corner2){
	assert(corner1.first >= 0);
	assert(corner1.second >= 0);
	assert(corner2.first >= 0);
	assert(corner2.second >= 0);

	ROWS = abs(corner2.second - corner1.second);
	COLS = abs(corner2.first - corner1.first);
	setIsMirrored(determineIfMirror(corner1, corner2));
	grid = new vector<vector<Tile*> > (ROWS, vector<Tile*>(COLS,NULL));
	int yPath = growsUp(corner1, corner2);
	if(isMirrored){
		for (int thisRow = 0; thisRow < ROWS; thisRow++) {
			for (int thisCol = 0; thisCol < COLS; thisCol++) {
				setOffsetTile(Tile::TileType::EMPTY, thisCol, thisRow,
					Coord2D(corner2.first+thisCol, corner2.second+yPath*thisRow));
			}
		}	
	}
	else{
		for (int thisRow = 0; thisRow < ROWS; thisRow++) {
			for (int thisCol = 0; thisCol < COLS; thisCol++) {
				setOffsetTile(Tile::TileType::EMPTY, thisCol, thisRow, 
					Coord2D(corner1.first+thisCol, corner1.second+yPath*thisRow));
			}
		}	

	}
}

Grid2D::Grid2D(const Grid2D& other) {
	ROWS = other.ROWS;
	COLS = other.COLS;
	isMirrored = other.isMirrored;

	grid = new vector<vector<Tile*> > (ROWS, vector<Tile*>(COLS,NULL));

	for (int thisRow = 0; thisRow < ROWS; thisRow++) {
		for(int thisCol = 0; thisCol < COLS; thisCol++) {
			Coord2D thisCoord2D = Coord2D(thisCol, thisRow);
			Tile* this_otherTile = other.getTile(thisCoord2D);
			setTile(this_otherTile->getType(), thisCoord2D);
		}
	}
}

Grid2D::Grid2D(Grid2D* other) {
	ROWS = other->ROWS;
	COLS = other->COLS;
	isMirrored = other->isMirrored;

	grid = new vector<vector<Tile*> >(ROWS, vector<Tile*>(COLS,NULL));
	for (int thisRow = 0; thisRow < ROWS; thisRow++) {
		for(int thisCol = 0; thisCol < COLS; thisCol++) {
			Coord2D thisCoord2D = Coord2D(thisCol, thisRow);
			Tile* this_otherTile = other->getTile(thisCoord2D);
			setTile(this_otherTile->getType(), thisCoord2D);
		}
	}
}

Grid2D::~Grid2D() {
    //delete grid;
    for(unsigned i = 0; i < grid->size(); i++) {
        for(unsigned j = 0; j < grid->at(i).size(); j++) {
            delete grid->at(i).at(j);
        }
    }
	delete grid;
}

std::string Grid2D::toString() {
	std::cout<<"DIM:"<<ROWS<<"X"<<COLS<<std::endl;
	std::string sb;
	// sb.resize((ROWS + 2) * (COLS + 3) + 1);

	// Top row border

	sb.append("*");
	for (int i = 0; i < COLS; i++)
		sb.append("-");
	sb.append("*");
	sb.append("\n");

	//Actual grid
	for (int thisRow = ROWS - 1; thisRow >= 0; thisRow--) {
		for (int thisCol = 0; thisCol < COLS; thisCol++) {
			
			// Left border
			if (thisCol == 0)
				sb.append("|");

			Tile* thisTile = (*grid)[thisRow][thisCol];
			sb.append(thisTile->getChar());
	
			// Right border
			if (thisCol == COLS - 1)
				sb.append("|");
		}
		sb.append("\n");
	}

	// Bottom row border
	sb.append("*");
	for (int i = 0; i < COLS; i++)
		sb.append("-");
	sb.append("*");
	sb.append("\n");

	return sb;
}

void Grid2D::setTile(Tile::TileType t, Coord2D location) {
	assertBounds(location);

	(*grid)[location.second][location.first] = new Tile(t, Coord2D(location));
}


//Basically setTile without the assert in order to create a mini grid
//preserving the coordinates of tiles.
void Grid2D::setOffsetTile(Tile::TileType t, int x, int y,  Coord2D location){

	(*grid)[x][y] = new Tile(t, Coord2D(location));
}

Tile* Grid2D::getTile(int x, int y) const {

	Coord2D temp(x, y);
    return getTile(temp);
}

Tile* Grid2D::getTile(Coord2D location) const {
	assertBounds(location);

	return (*grid)[location.second][location.first];
}

void Grid2D::assertBounds(Coord2D location) const {
	assert(checkBounds(location));
}

bool Grid2D::checkBounds(Coord2D location) const{
	int x = location.first;
	int y = location.second;

	// Make sure they aren't negative
	if (x < 0 || y < 0) return false;

	return x < COLS && y < ROWS;
}

bool Grid2D::determineIfMirror(Coord2D c1, Coord2D c2){
	bool result = false;
	if(c2.first - c1.first < 0){
		result = true;		
	}
	return result;
	
}

int Grid2D::growsUp(Coord2D c1, Coord2D c2){
	int result = -1;
	if(c2.second - c1.second < 0){
		result = 1;
	}
	return result;	
}

Coord2D Grid2D::getGridDimensions() {
	return  Coord2D(COLS, ROWS);
}

int Grid2D::size() {
	return ROWS * COLS;
}

std::string Grid2D::getChar(Coord2D location) {
	return getTile(location)->getChar();
}

bool Grid2D::canGoUp(Coord2D location) {
	return location.second < ROWS - 1;
}

bool Grid2D::canGoDown(Coord2D location) {
	return location.second > 0;
}

bool Grid2D::canGoLeft(Coord2D location) {
	return location.first > 0;
}

bool Grid2D::canGoRight(Coord2D location) {
	return location.first < COLS - 1;
}

Tile* Grid2D::getUp(Coord2D fromHere) {
	if (!canGoUp(fromHere)) return NULL;

	return getTile( Coord2D(fromHere.first, fromHere.second + 1));
}

Tile* Grid2D::getDown(Coord2D fromHere) {
	if (!canGoDown(fromHere)) return NULL;

	return getTile( Coord2D(fromHere.first, fromHere.second - 1));
}

Tile* Grid2D::getLeft(Coord2D fromHere) {
	if (!canGoLeft(fromHere)) return NULL;

	return getTile( Coord2D(fromHere.first - 1, fromHere.second));
}

Tile* Grid2D::getRight(Coord2D fromHere) {
	if (!canGoRight(fromHere)) return NULL;

	return getTile( Coord2D(fromHere.first + 1, fromHere.second));
}

/**
 * Mark a STRAIGHT line on this grid between the two points.
 * Points must be within bounds of this grid.
 * @param point1 an endpoint
 * @param point2 another endpoint
 */
void Grid2D::markLine(Coord2D point1, Coord2D point2, bool mark) {
	assertBounds(point1);
	assertBounds(point2);

	assert(point1.first == point2.first || point1.second == point2.second);

	if (point1 == point2) {
		Tile* t = getTile(point1);
		t->setMark(mark);
		return;
	}

	// If on the same row
	if (point1.second == point2.second) {
		for (int i = 0; i < COLS; i++) {
			Tile* thisTile = getTile(Coord2D(i, point1.second));
			thisTile->setMark(mark);
		}
	}

	// Else, they're on the same column
	else {
		for (int i = 0; i < ROWS; i++) {
			Tile* thisTile = getTile(Coord2D(point1.first, i));
			thisTile->setMark(mark);
		}
	}
}

void Grid2D::setTypeLine(Coord2D point1, Coord2D point2, Tile::TileType type, bool prioritize) {
	assertBounds(point1);
	assertBounds(point2);

	assert(point1.first == point2.first || point1.second == point2.second);

	if (point1 == point2) {
		Tile* t = getTile(point1);
		if (prioritize || t->getType() == Tile::TileType::EMPTY)
			t->setType(type);
		return;
	}

	// If on the same row
	if (point1.second == point2.second) {

		// Iterate through from least x to greatest x,
		// whichever is which
		for (int i = (point1.first <= point2.first ? point1.first : point2.first);
		i <= (point1.first >  point2.first ? point1.first : point2.first);
		i++) {

			Tile* thisTile = getTile(Coord2D(i, point1.second));

			if (prioritize || thisTile->getType() == Tile::TileType::EMPTY)
				thisTile->setType(type);
		}
	}

	// Else, they're on the same column
	else {

		// Iterate through from least y to greatest y,
		// whichever is which
		for (int i = (point1.second <= point2.second ? point1.second : point2.second);
		i <= (point1.second >  point2.second ? point1.second : point2.second);
		i++) {

		Tile* thisTile = getTile(Coord2D(point1.first, i));

		if (prioritize || thisTile->getType() == Tile::TileType::EMPTY)
			thisTile->setType(type);
		}
	}
}

void Grid2D::setIsMirrored(bool value){
	this->isMirrored = value;
}

bool Grid2D::checkIsMirrored(){
	return this->isMirrored;
}

void Grid2D::setTypeRect(Coord2D lowerLeft, Coord2D upperRight, Tile::TileType type, bool prioritize) {
	assertBounds(lowerLeft);
	assertBounds(upperRight);

	assert(lowerLeft.first <= upperRight.first);
	assert(lowerLeft.second <= upperRight.second);

	if (lowerLeft.first == upperRight.first || lowerLeft.second == upperRight.second) {
		setTypeLine(lowerLeft, upperRight, type, prioritize);
		return;
	}

	// If we're here, then we're marking a non-line rectangle,
	// and the arguments were provided in correct order
	for (int thisY = lowerLeft.second; thisY <= upperRight.second; thisY++) {

		// Mark row by row

		Coord2D thisRowLeft = Coord2D(lowerLeft.first, thisY);
		Coord2D thisRowRight = Coord2D(upperRight.first, thisY);

		setTypeLine(thisRowLeft, thisRowRight, type, prioritize);
	}
}

void Grid2D::setTypeLine(Coord2D point1, Coord2D point2, Tile::TileType type, int layers, bool prioritize) {
	for (int thisLevel = 0; thisLevel <= layers; thisLevel++) {

		// Row (horizontal)
		if (point1.second == point2.second) {
			// Do row on top, offset by thisLevel
			Coord2D point1Layered = Coord2D(point1.first, point1.second + thisLevel);
			Coord2D point2Layered = Coord2D(point2.first, point2.second + thisLevel);

			if (checkBounds(point1Layered) && checkBounds(point2Layered))
				setTypeLine(point1Layered, point2Layered, type, prioritize);

			// Do row on bot, offset by thisLevel
			point1Layered =  Coord2D(point1.first, point1.second - thisLevel);
			point2Layered =  Coord2D(point2.first, point2.second - thisLevel);

			if (checkBounds(point1Layered) && checkBounds(point2Layered))
				setTypeLine(point1Layered, point2Layered, type, prioritize);
		}

		// Col (vertical)
		else if (point1.first == point2.first) {
 
			// Do col on left, offset by thisLevel
			Coord2D point1Layered =  Coord2D(point1.first - thisLevel, point1.second);
			Coord2D point2Layered =  Coord2D(point2.first - thisLevel, point2.second);

			if (checkBounds(point1Layered) && checkBounds(point2Layered))
				setTypeLine(point1Layered, point2Layered, type, prioritize);

			// Do col on right, offset by thisLevel
			point1Layered =  Coord2D(point1.first + thisLevel, point1.second);
			point2Layered =  Coord2D(point1.first + thisLevel, point2.second);

			if (checkBounds(point1Layered) && checkBounds(point2Layered))
				setTypeLine(point1Layered, point2Layered, type, prioritize);
		}

		else
			assert(false);
	}
}

void Grid2D::markRect(Coord2D lowerLeft, Coord2D upperRight, bool mark) {
	assertBounds(lowerLeft);
	assertBounds(upperRight);

	assert(lowerLeft.first <= upperRight.first);
	assert(lowerLeft.second <= upperRight.second);

	if (lowerLeft.first == upperRight.first || lowerLeft.second == upperRight.second) {
		markLine(lowerLeft, upperRight, mark);
		return;
	}

	// If we're here, then we're marking a non-line rectangle,
	// and the arguments were provided in correct order
	for (int thisY = lowerLeft.second; thisY <= upperRight.second; thisY++) {

		// Mark row by row
		Coord2D thisRowLeft =  Coord2D(lowerLeft.first, thisY);
		Coord2D thisRowRight =  Coord2D(upperRight.first, thisY);

		markLine(thisRowLeft, thisRowRight, mark);
	}
}

// Tile* Grid2D::getTraversableNeighbors(Coord2D location) {
// 	Tile neighbors[4] ;
// 	if(canGoUp(location)) {
// 		Tile* upNeighbor = getUp(location);
// 		if(upNeighbor->getType() != Tile::TileType::NON_TRAVERSABLE) {
// 			neighbors[0]
// 		}
// 	}
// }

std::set<Tile*> Grid2D::getTraversableNeighbors(Coord2D location) {
	std::set<Tile*> neighbors;

	if (canGoUp(location)) {
		Tile* upNeighbor = getUp(location);

		if (upNeighbor->getType() != Tile::TileType::NON_TRAVERSABLE)
			neighbors.insert(upNeighbor);
	}

	if (canGoDown(location)) {
		Tile* downNeighbor = getDown(location);

		if (downNeighbor->getType() != Tile::TileType::NON_TRAVERSABLE)
			neighbors.insert(downNeighbor);
	}

	if (canGoLeft(location)) {
		Tile* leftNeighbor = getLeft(location);

		if (leftNeighbor->getType() != Tile::TileType::NON_TRAVERSABLE)
			neighbors.insert(leftNeighbor);
	}

	if (canGoRight(location)) {
		Tile* rightNeighbor = getRight(location);

		if (rightNeighbor->getType() != Tile::TileType::NON_TRAVERSABLE)
			neighbors.insert(rightNeighbor);
	}

	return neighbors;
}

int Grid2D::getROWS(){
	return ROWS;
}

int Grid2D::getCOLS(){
	return COLS;
}

bool Grid2D::areNeighbors(Coord2D c1, Coord2D c2){
    bool result = false;
    if(abs(c2.first - c1.first) == 1 || abs(c2.second - c1.second) == 1)
        result = true;

   return result;
}
