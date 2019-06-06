#include "Tile.h"

using namespace std;

Tile::Tile() {
    this->marked=false;
    this->type=TileType::EMPTY;
    this->distance=INT_MAX; // FIXME: does this work?
                            // after adding <climits> it should! - Signed: Andrew
    this->prev=NULL;
    this->location=NULL;
}

Tile::Tile(TileType type) {
    this->marked=false;
    this->type=type;
    this->distance=INT_MAX; 
    this->prev=NULL;
    this->location=NULL;
}

Tile::Tile(TileType type, Coord2D location) {
    this->marked=false;
    this->type=type;
    this->distance=INT_MAX; 
    this->prev=NULL;
    this->location=new Coord2D(location); // FIXME: dont forget to delete
                                            // done - Signed: Andrew
}

Tile::Tile(TileType type, bool mark) {
    this->marked=mark;
    this->type=type;
    this->distance=INT_MAX;
    this->prev=NULL;
    this->location=NULL;
}

Tile::Tile(const Tile& t) {
    this->marked=t.marked;
    this->type=t.type;
    this->distance=t.distance; // java file has this INT_MAX
    this->prev=t.prev; // java file has this NULL
    this->location=new Coord2D(*(t.location));
}

Tile::~Tile() {
    delete location;
}

std::string Tile::getChar() const {
    if (isMarked()) {
        return "*";
    }
    switch (type)
    {
    case EMPTY:
        return ".";
        break;
    case TRAVERSABLE:
        return "t";
        break;
    case NON_TRAVERSABLE:
        return "N";
        break;
    default:
        return "?";
        break;
    }
}

Tile::TileType Tile::getType() const {
    return type;
}
bool Tile::isMarked() const {
    return marked;
}

void Tile::setMark(bool mark) {
    this->marked=mark;
}

void Tile::setType(TileType t) {
    this->type=t;
}

string Tile::toString() const {
    // returns string: "TileType <type>, <_ OR not> marked, distance = <distance>, location <location>"
    string typestr = (string[]){"EMPTY","TRAVERSABLE","NON_TRAVERSABLE"}[this->type];
    // string info = "";
    // info.append("TileType ").append(typestr).append(", ").append(marked ? "":"not ").append(
    //     "marked, distance = ").append(distance).append(", location ").append(location.toString()); // does location have a custom toString defined?
    // return info;
    return "TileType " + typestr + ", " + (marked ? "": "not ") + "marked, distance = " + std::to_string(distance) + ", location " + (location ? coord_to_string(*location) : "NULL");
}

void Tile::setDistance(int distance) {   
    this->distance = distance;
}

int Tile::getDistance() const {
    return distance;
}

void Tile::setPreviousTile(Tile* prev) {
    this->prev = prev;
}

Tile* Tile::getPreviousTile() const {
    return prev;
}

Coord2D* Tile::getLocation() const {    
    return location;
}