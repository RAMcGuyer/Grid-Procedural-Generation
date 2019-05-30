
#ifndef __COORD2D_H__
#define __COORD2D_H__
#include <stdlib.h>
#include <string>

public class Coord2D {
//Coord2D is a coordinate object that enables us to keep track of
// which tiles in our grid our landmarks.
    private int x;
    private int y;

//Constructors, take in an X and a Y coordinate.
    public Coord2D(int x, int y) {

        this.x = x;
        this.y = y;
    }
// Constructor that creates a deep copy of a coordinate object
    public Coord2D(Coord2D other) {
        
        this.x = other.x;
        this.y = other.y;
    }
//Getters    
    public int getX() {
        return x;
    }
    
    public int getY() {
        return y;
    }

//Displays contents of object to the console.
    public string toString() {
        
        return "{" + Integer.toString(x) + ", " + Integer.toString(y) + "}";
    }

//Comparator function    
    public bool equals(Object other) {
        
//        return this.x == other.x && this.y == other.y;
        if (other instanceof Coord2D) {
            
            Coord2D otherCoord = (Coord2D) other;
            
            return this.x == otherCoord.x && this.y == otherCoord.y;
        }
        
        else return false;
    }

//For use in the hashset
    public int hashCode() {
        
        return x + y;
    }
    
}
#endif
