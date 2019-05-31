
#ifndef __COORD2D_H__
#define __COORD2D_H__
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
        //Constructors, take in an X and a Y coordinate.
        Coord2D(int x, int y) {
            this->x = x;
            this->y = y;
        }
        // Constructor that creates a deep copy of a coordinate object
        Coord2D(const Coord2D& other) {
            this->x = other.x;
            this->y = other.y;
        }
        //Getters    
        int getX() const {
            return x;
        }
        
        int getY() const {
            return y;
        }

        //Displays contents of object to the console.
        std::string toString() {
            return "{" + std::to_string(x) + ", " + std::to_string(y) + "}";
        }

        //Comparator function   
        template <class T>
        bool equals(T other) {    
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
        int hashCode() const {
            return x + y;
        }
};
#endif
