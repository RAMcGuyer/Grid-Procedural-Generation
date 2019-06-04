#include "Coord2D.h"

Coord2D::Coord2D(){
    this->x = 0;
    this->y = 0;
}

//Constructors, take in an X and a Y coordinate.
Coord2D::Coord2D(int x, int y) {
    this->x = x;
    this->y = y;
}
// Constructor that creates a deep copy of a coordinate object
Coord2D::Coord2D(const Coord2D& other) {
    this->x = other.x;
    this->y = other.y;
}

Coord2D::~Coord2D(){
//This is a deconstructor
}
//Getters    
int Coord2D::getX() const {
    return this->x;
}

int Coord2D::getY() const {
    return this->y;
}

//Displays contents of object to the console.
std::string Coord2D::toString() {
    return "{" + std::to_string(x) + ", " + std::to_string(y) + "}";
}

//For use in the hashset
int Coord2D::hashCode() const {
    return x + y;
}

void Coord2D::operator=(const Coord2D& rhs) {
    this->x = rhs.x;
    this->y = rhs.y;
}

bool Coord2D::operator==(const Coord2D& rhs) {
    return this->x == rhs.x && this->y == rhs.y;
}
