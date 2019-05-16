/**
 * Class used to represent tiles, which can be marked or unmarked.
 * @author Ryan Villena
 *
 */
public class Tile {
    
    public enum TileType {
        EMPTY,
        TRAVERSABLE,
        NON_TRAVERSABLE
    }

    Tile() {
        this.marked = false;
        this.type = TileType.EMPTY;
        this.distance = Integer.MAX_VALUE;
        this.prev = null;
    }

    Tile(TileType type) {
        
        this.marked = false;
        this.type = type;
        this.distance = Integer.MAX_VALUE;
        this.prev = null;
        this.location = null;
    }
    
    Tile(TileType type, Coord2D location) {
        
        this.marked = false;
        this.type = type;
        this.distance = Integer.MAX_VALUE;
        this.prev = null;
        this.location = new Coord2D(location);
    }
    
    Tile(TileType type, int distance) {
        
        this.marked = false;
        this.type = type;
        this.distance = distance;
        this.prev = null;
    }
    
    Tile(TileType type, boolean mark) {
        
        this.marked = mark;
        this.type = type;
        this.distance = Integer.MAX_VALUE;
        this.prev = null;
    }
    
    Tile(Tile t) {
        this.marked = t.marked;
        this.type = t.type;
        this.distance = Integer.MAX_VALUE;
        this.prev = null;
    }
    
    public char getChar() {
        
        if (isMarked())
            return '*';
        
        switch (type) {
            
            case EMPTY:
                return '.';
            case TRAVERSABLE:
                return 't';
            case NON_TRAVERSABLE:
                return 'N';
            default:
                return '?';
        }
    }
    
    public boolean isMarked() {
        return marked;
    }
    
    public void setMark(boolean mark) {
        this.marked = mark;
    }
    
    public void setType(TileType t) {
        
        this.type = t;
    }
    
    public TileType getType() {
        
        return type;
    }
    
    @Override
    public String toString() {
        
        return TileType.class.getSimpleName() + " " + this.type.toString() + ", "
                + (marked ? "" : "not ") + "marked, "
                + "distance = " + (distance == Integer.MAX_VALUE ? "MAX_VALUE" : distance) + ", "
                + "location " + location.toString();
    }
    
    public void setDistance(int distance) {
        
        this.distance = distance;
    }
    
    public int getDistance() {
        
        return distance;
    }
    
    public void setPreviousTile(Tile prev) {
        
        this.prev = prev;
    }
    
    public Tile getPreviousTile() {
        
        return prev;
    }
    
    public Coord2D getLocation() {
        
        return location;
    }
    
    private boolean marked;
    private TileType type;
    private int distance;
    private Tile prev;
    private Coord2D location;
}