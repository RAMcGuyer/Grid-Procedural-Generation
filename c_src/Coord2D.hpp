#ifndef __COORD2D_HPP__
#define __COORD2D_HPP__

#include <stdlib.h> 
#include <string>
#include <typeinfo>

class Coord2D {
//Coord2D is a coordinate object that enables us to keep track of
// which tiles in our grid our landmarks.
    private: 
        int x;
        int y;

    public:
        Coord2D();
        Coord2D(int x, int y);
        Coord2D(const Coord2D& other);
        int getX() const;
        int getY() const;
        std::string toString();
        template <class T> bool equals(T other);
        int hashCode() const;
        void operator=(const Coord2D& rhs);
        bool operator==(const Coord2D& rhs);
};

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

//Comparator function   
template <class T>
bool Coord2D::equals(T other) {    
    //return this.x == other.x && this.y == other.y;
    if ( (typeid(other)==typeid(Coord2D)) ) {
        Coord2D otherCoord = (Coord2D) other;
        return this->x == otherCoord.x && this->y == otherCoord.y;
    }
    else{ 
        return false;
    }
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

#endif
