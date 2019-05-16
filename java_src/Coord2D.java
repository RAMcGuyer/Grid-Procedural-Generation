
public class Coord2D {

    public Coord2D(int x, int y) {

        this.x = x;
        this.y = y;
    }

    public Coord2D(Coord2D other) {
        
        this.x = other.x;
        this.y = other.y;
    }
    
    public int getX() {
        return x;
    }
    
    public int getY() {
        return y;
    }

    @Override
    public String toString() {
        
        return "{" + Integer.toString(x) + ", " + Integer.toString(y) + "}";
    }
    
    @Override
    public boolean equals(Object other) {
        
//        return this.x == other.x && this.y == other.y;
        if (other instanceof Coord2D) {
            
            Coord2D otherCoord = (Coord2D) other;
            
            return this.x == otherCoord.x && this.y == otherCoord.y;
        }
        
        else return false;
    }
    
    @Override
    public int hashCode() {
        
        return x + y;
    }
    
    private int x;
    private int y;
}
