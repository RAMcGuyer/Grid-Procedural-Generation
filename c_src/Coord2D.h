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
        Coord2D();
        Coord2D(int x, int y);
        Coord2D(const Coord2D& other);
        ~Coord2D();
        int getX() const;
        int getY() const;
        std::string toString();
        int hashCode() const;
        void operator=(const Coord2D& rhs);
        bool operator==(const Coord2D& rhs);

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
};


#endif //__COORD2D_H__
