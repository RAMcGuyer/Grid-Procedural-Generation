import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

public class Path {

    public Path(Grid2D grid) {
        
        this.grid = grid;
        joints = new ArrayList<Coord2D>();
        this.thickness = 0;
    }
    
    public Path(Grid2D grid, Coord2D point1, Coord2D point2, int thickness) {
        
        this.grid = grid;
        this.joints = new ArrayList<Coord2D>();
        this.thickness = thickness;
        
        populateBestPath(point1, point2);
    }
    
    public Path(Grid2D grid, List<Coord2D> joints, int thickness) {
        
        this.grid = grid;
        this.joints = new ArrayList<Coord2D>(joints);
        this.thickness = thickness;
    }
    
    public boolean areCompatibleJoints(Coord2D joint1, Coord2D joint2) {
        
        return joint1.getX() == joint2.getX() || joint1.getY() == joint2.getY();
    }
    
    /**
     * Append a joint to this path if it's compatible.
     * For more information, see the other overload for this function,
     * addJoint(Coord2D, int).
     * @param newJoint Joint to be added
     * @return True if this joint was compatible and added successfully, false otherwise
     */
    public boolean addJoint(Coord2D newJoint) {
        
        if (joints.isEmpty()) {
            
            joints.add(newJoint);
            return true;
        }
        
        else {
            
            return addJoint(newJoint, joints.size());
        }   
    }
    
    /**
     * Add a joint to this path at the specified index.
     * Joints are required to be compatible,
     * i.e. they lie on the same line as their adjacent joint.
     * If you try to use this function and add a joint which isn't compatible to its neighbors,
     * this function returns false and does NOT add the joint.
     * Else, it adds the joint to this path and returns true.
     * Don't worry if this Path isn't yet complete (i.e. doesn't have 2 or more points);
     * you can add joints with this method and still return true safely.
     * @param newJoint Joint to be added
     * @param index Index in this list of joints
     * @return True if this joint was compatible and added successfully, false otherwise
     */
    public boolean addJoint(Coord2D newJoint, int index) {
        
        joints.add(index, newJoint);
        

        ListIterator<Coord2D> it = joints.listIterator(index);
        
        // Check compatibility with joint before, if it exists
        if (it.hasPrevious()) {
            
            Coord2D left_neighbor = it.previous();
            
            if (!areCompatibleJoints(left_neighbor, newJoint)) {
                
                joints.remove(index);
                return false;
            }
            
            // "Reset" iterator
            it.next();
        }
        
        // Check compatibility with joint next, if it exists
        if (it.hasNext()) {
            
            Coord2D right_neighbor = it.next();
            
            if (!areCompatibleJoints(newJoint, right_neighbor)) {
                
                joints.remove(index);
                return false;
            }
            // no need to reset iterator here
        }
        
        // We've checked its compatibility with both neighbors,
        // so nothing else to do but return true
        return true;
        
    }
    
    /**
     * Sets the type of tiles on this path.
     * Should be used to either set a path as traversable,
     * or to "erase" by setting them to empty (or something else!).
     * Note that you must first set this object's path by giving it 2 or more joints,
     * otherwise you're trying to draw an ill-defined path
     * and I'm going to abort your program.
     * @param type Type that all tiles on this path should be set to
     */
    public void setPathType(Tile.TileType type, boolean prioritize) {
        
        assert joints.size() >= 2 : " Not enough joints in this path,\n"
            + "here are all joints: " + joints.toString();
        
        Coord2D firstJoint = joints.get(0);
        ListIterator<Coord2D> it = joints.listIterator();
        
        while (it.hasNext()) {
            
            Coord2D secondJoint = it.next();
            grid.setTypeLine(firstJoint, secondJoint, type, thickness, prioritize);            
            
            // Slide first joint
            firstJoint = secondJoint;
        }
    }
    
    
    
    /**
     * Populates this Path's "joints" list with the best path between point1 and point2.
     * This is accomplished by implementing Djikstra's algorithm.
     * This Path's grid and joints MUST be initialized before this method is called.
     * Algorithm adapted from the pseudocode found here:
     * https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Pseudocode
     * @param src The coordinate that the path should start from
     * @param dest The coordinate that the path should lead to
     */
    private void populateBestPath(Coord2D src, Coord2D dest) {
        
        
        Grid2D tempGrid = new Grid2D(this.grid); // generates copy, maintaining types
        
        Tile srcTile  = tempGrid.getTile(src);
        Tile destTile = tempGrid.getTile(dest);
        srcTile.setDistance(0);
        
        // Do some error checking 
        assert src != dest : " Attempted autopath to the same tile";
        assert srcTile.getType() != Tile.TileType.NON_TRAVERSABLE
                : " Path attempted on non-traversable tile " + srcTile.toString();
        assert destTile.getType() != Tile.TileType.NON_TRAVERSABLE
                : " Path attempted on non-traversable tile " + destTile.toString();
        
        
        
        // Populate set Q
        Set<Tile> setQ = new HashSet<Tile>();
        setQ.add(grid.getTile(new Coord2D(0, 0)));
        for (Tile t : tempGrid) {
            
            if (t.getType() != Tile.TileType.NON_TRAVERSABLE)
                setQ.add(t);
        }

        assert setQ.contains(srcTile)  : " setQ doesn't contain srcTile";
        assert setQ.contains(destTile) : " setQ doesn't contain destTile";
        
        Tile uTile = null;
        
        while (!setQ.isEmpty()) {
            
            // Get tile with minimum distance from setQ
            int runningMin = Integer.MAX_VALUE;
            for (Tile t : setQ) {
                
                if (t.getDistance() < runningMin) {
                    
                    runningMin = t.getDistance();
                    uTile = t;
                }
            }
            
            // Make sure uTile is properly set,
            // then remove it from setQ
            assert uTile != null : " Minimum distance tile uTile not properly set";
            assert setQ.contains(uTile) : " setQ doesn't contain uTile " + uTile.toString();
            setQ.remove(uTile);
            
            // Break out if we've reached the destination,
            // we need to now construct the path via reverse iteration
            if (uTile == destTile)
                break;
            
            // Update distances of all uTile's current neighbors            
            Set<Tile> uNeighbors = tempGrid.getTraversableNeighbors(uTile.getLocation());
            
            for (Tile thisNeighbor : uNeighbors) {
                
                int currentDist = uTile.getDistance() + 1;
                
                if (currentDist < thisNeighbor.getDistance()) {
                    
                    thisNeighbor.setDistance(currentDist);
                    thisNeighbor.setPreviousTile(uTile);
                }
            }
        }
        
        
        
        // Ensure that uTile is actually usable
        assert uTile.getPreviousTile() != null || uTile == srcTile
                : " Condition specified by Dijkstra's not met";
            
        // Populate joints by backtracing
        while (uTile != null) {
            
            joints.add(uTile.getLocation());
            uTile = uTile.getPreviousTile();
            
            assert (!(uTile == null && joints.size() < 2))
                : "Not enough prev's? For sure not enough joints\n"
                + "Perhaps src and dest are the same?\n"
                + "src:  " + src.toString() + '\n'
                + "dest: " + dest.toString() + '\n'
                + "src.equals(dest)? " + src.equals(dest);
        }
    }
    
    
    
    private Grid2D grid;
    private List<Coord2D> joints;
    private int thickness;
}
