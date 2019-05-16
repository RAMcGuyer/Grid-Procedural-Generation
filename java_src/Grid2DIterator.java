import java.util.Iterator;
import java.util.NoSuchElementException;

public class Grid2DIterator implements Iterator<Tile> {

    public Grid2DIterator(Grid2D g) {
        
        this.g = g;
        current = new Coord2D(0, 0);
    }
    
    @Override
    public boolean hasNext() {
        
        int xMax = g.getGridDimensions().getX();
        int yMax = g.getGridDimensions().getY();
        
        return (current.getX() < xMax - 1) || (current.getY() < yMax - 1);
    }

    @Override
    public Tile next() {
        
        if (!hasNext()) throw new NoSuchElementException();
        
        // Iterate through row first, if possible
        if (current.getX() < g.getGridDimensions().getX() - 1) {
            
            current = new Coord2D(current.getX() + 1, current.getY());
        }
        
        // If we're at the end of a row, progress thru columns and reset the row
        else if (current.getX() == g.getGridDimensions().getX() - 1) {
            
            current = new Coord2D(0, current.getY() + 1);
        }
        
        else {
            
            assert false : " Invalid iterator coordinate " + current.toString();
        }
        
        return g.getTile(current);
    }
    
    private Grid2D g;
    private Coord2D current;

}
