#include <string>
using namespace std;

class Tile public {
    enum TileType{
        EMPTY,
        TRAVERSABLE,
        NON_TRAVERSABLE
    };

    private:
        bool marked;
        TileType type;
        int distance;
        Tile prev;
        Coord2D location;

    public:
        Tile() {
            this->marked=false;
            this->type=TileType::EMPTY;
            this->distance=INT_MAX; // FIXME: does this work?
            this->prev=NULL;
        }
        Tile(TileType type) {
            this->marked=false;
            this->type=type;
            this->distance=INT_MAX; 
            this->prev=NULL;
            this->location=NULL;
        }
        Tile(TileType type, Coord2D location) {
            this->marked=false;
            this->type=type;
            this->distance=INT_MAX; 
            this->prev=NULL;
            this->location=new Coord2D(location); // FIXME: dont forget to delete
        }
        Tile(TileType type, bool mark) {
            this->marked=mark;
            this->type=type;
            this->distance=INT_MAX;
            this->prev=NULL;
        }
        Tile(Tile t) {
            this->marked=t.marked;
            this->type=t.type;
            this->distance=INT_MAX;
            this->prev=NULL;
        }

        char getChar() {
            if (isMarked()) {
                return '*';
            }
            switch (type)
            {
            case EMPTY:
                return '.';
                break;
            case TRAVERSABLE:
                return 't';
                break;
            case NON_TRAVERSABLE:
                return 'N';
                break;
            default:
                return '?';
                break;
            }
        }

        TileType getType() {
            return type;
        }
        bool isMarked() {
            return marked;
        }

        void setMark(bool mark) {
            this->marked=mark;
        }

        void setType(TileType t) {
            this->type=t;
        }

        string toString() {
            // returns string: "TileType <type>, <_ OR not> marked, distance = <distance>, location <location>"
            string type = (string[]){"EMPTY","TRAVERSABLE","NON_TRAVERSABLE"}[this->type];
            string info;
            // string info = "TileType " + type + ", " + (marked ? "":"not ") + "marked, distance" 
            return info;
        }

        void setDistance(int distance) {   
            this.distance = distance;
        }
        
        int getDistance() {
            return distance;
        }
        
        void setPreviousTile(Tile prev) {
            this->prev = prev;
        }
        
        Tile getPreviousTile() {
            return prev;
        }
        
        Coord2D getLocation() {    
            return location;
        }
};