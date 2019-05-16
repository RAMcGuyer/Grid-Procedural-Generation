import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Project to demonstrate procedural generation on a 2D grid.
 */

/**
 * Driver class to run tests/demos on procedural generation,
 * or even just user tests.
 * @author Ryan Villena
 *
 */
public class MainDriver {

    public static void testSetOfPoints() {
        
        Coord2D original = new Coord2D(3, 4);
        Coord2D duplicate = new Coord2D(original);
        Coord2D anotherone = new Coord2D(6, 9);
        
        assert original != duplicate : "original == duplicate";
        assert duplicate.equals(original) : "Coord2D.equals() broken";
        
        Set<Coord2D> set = new HashSet<Coord2D>(3);
        set.add(original);
        System.out.println("Set contains original. Contains duplicate? " + set.contains(duplicate));
//        if (!set.contains(duplicate))
            set.add(duplicate);
        set.add(anotherone);
        
        List<Coord2D> list = new ArrayList<Coord2D>(set.size());
        list.addAll(set);
        
        System.out.println("Set:  " + set);
        System.out.println("List: " + list);
    }
    
    public static void testGrid() {
        
        Coord2D gridDimensions = new Coord2D(7, 13);
        Coord2D testPoint      = new Coord2D(0, gridDimensions.getY() - 1);
        Coord2D emptyPoint     = new Coord2D(3, 10);
        @SuppressWarnings("unused")
        Coord2D failPoint      = new Coord2D(7, 14);
        
        Grid2D grid = new Grid2D(gridDimensions);
        grid.setTile(Tile.TileType.TRAVERSABLE, testPoint);
        
        // Test tile types
        System.out.println(grid.toString());
        System.out.println("Tile at " + testPoint.toString()  + ": " + grid.getTile(testPoint) );
        System.out.println("Tile at " + emptyPoint.toString() + ": " + grid.getTile(emptyPoint));
//        System.out.println("Tile at " + failPoint.toString()  + ": " + grid.getTile(failPoint));
        
        // Test directions
        System.out.println("Can testPoint go up? "   + grid.canGoUp(testPoint));
        System.out.println("Can testPoint go down? " + grid.canGoDown(testPoint));
        System.out.println("Can testPoint go left? " + grid.canGoLeft(testPoint));
        System.out.println("Can testPoint go right? "+ grid.canGoRight(testPoint));
        
        System.out.println("Getting the up of testPoint:" + grid.getUp(testPoint));
        System.out.println("Getting testPoint manually: " + grid.getTile(testPoint));
        
        System.out.println(grid);
        
        // Test marks
//        Coord2D markPoint = new Coord2D(5, 5);
//        Coord2D unmarkedPoint = new Coord2D(5, 4);
//        grid.getTile(markPoint).setMark(true);
//        
//        System.out.println("Just marked tile " + markPoint + ", should be marked: " + grid.getTile(markPoint));
//        System.out.println("Testing unmarked tile " + unmarkedPoint + ", should not be marked: " + grid.getTile(unmarkedPoint));
//        System.out.println(grid);
    }
    
    public static void testMarkRow() {
        
        Coord2D gridDimensions = new Coord2D(10, 10);
        Grid2D grid = new Grid2D(gridDimensions);
        
        System.out.println("Empty grid:\n" + grid);
        
        Coord2D lowbar_left = new Coord2D(0, 2);
        Coord2D lowbar_right = new Coord2D(gridDimensions.getX() - 1, lowbar_left.getY());

        
        System.out.println("Marking lower bar...");
        grid.setTypeLine(lowbar_left, lowbar_right, Tile.TileType.TRAVERSABLE, 0, true);
        System.out.println(grid);
        
        Coord2D vertbar_down = new Coord2D(2, 0);
        Coord2D vertbar_up = new Coord2D(vertbar_down.getX(), gridDimensions.getY() - 1);
        
        System.out.println("Marking vert bar...");
        grid.setTypeLine(vertbar_down, vertbar_up, Tile.TileType.TRAVERSABLE, 2, true);
        System.out.println(grid);
        
        
    }
    
    public static void testMarkRect() {
        
        Coord2D gridDimensions = new Coord2D(7, 13);
        Grid2D grid = new Grid2D(gridDimensions);
        
        System.out.println("Empty grid:\n" + grid);
        
        Coord2D middleBand_lowleft = new Coord2D(0, 3);
        Coord2D middleBand_upRight = new Coord2D(gridDimensions.getX() - 1, 6);
        
        System.out.println("Marking a band in the middle:");
        grid.markRect(middleBand_lowleft, middleBand_upRight, true);
        
        System.out.println(grid);
    }
    
    public static void testPaths() {
        
        Grid2D grid = new Grid2D(new Coord2D(50, 50));
        
        System.out.println("Empty grid:\n\n" + grid);
        
//        List<Coord2D> pathJoints = new ArrayList<Coord2D>();
////        pathJoints.add(new Coord2D(2, 3));
////        pathJoints.add(new Coord2D(10, 3));
//        pathJoints.add(new Coord2D(10, 1));
//        pathJoints.add(new Coord2D(10, 45));
//        pathJoints.add(new Coord2D(48, 45));
//        pathJoints.add(new Coord2D(48, 2));
//        
//        Path path = new Path(grid, pathJoints);
////        grid.setTypeLine(pathJoints.get(0), pathJoints.get(1), Tile.TileType.TRAVERSABLE);
//        path.setPathType(Tile.TileType.TRAVERSABLE);
//        
//        System.out.println("Populated grid:\n\n" + grid);
        
        Path path = new Path(grid);
        assert path.addJoint(new Coord2D(10, 1));
        assert path.addJoint(new Coord2D(10, 2));
        assert path.addJoint(new Coord2D(30, 2));
        assert path.addJoint(new Coord2D(30, 0));
        
        path.setPathType(Tile.TileType.TRAVERSABLE, true);
        

        
        Grid2D copy = new Grid2D(grid);
        copy.setTile(Tile.TileType.TRAVERSABLE, new Coord2D(0, 0));
        
        System.out.println("Populated grid:\n\n" + grid);
        System.out.println("Copy:\n\n" + copy);
    }
       
    public static void testIterator() {
        
        Coord2D gridDimensions = new Coord2D(50, 50);
        Grid2D grid = new Grid2D(gridDimensions);
        
        Set<Tile> tiles = new HashSet<Tile>();
        tiles.add(grid.getTile(new Coord2D(0, 0))); // careful, need to manually add first element
        for (Tile t : grid) {
            
            tiles.add(t);
        }
        
        System.out.println("Expected size: " + (gridDimensions.getX() * gridDimensions.getY()));
        System.out.println("Num of tiles: " + tiles.size());
    }
    
    public static void testDijkstra() {
        
        Coord2D gridDimensions = new Coord2D(10, 10);
        Grid2D grid = new Grid2D(gridDimensions);
        
        // Make a non-traversable obstacle
        Coord2D obstacle_LowerLeft = new Coord2D(5, 2);
        Coord2D obstacle_UpperRight= new Coord2D(7, 5);
        
        grid.setTypeRect(obstacle_LowerLeft, obstacle_UpperRight, Tile.TileType.NON_TRAVERSABLE, true);
        
        System.out.println("Grid with single obstacle:\n\n" + grid);
        
        Coord2D pointA = new Coord2D(1, 1);
        Coord2D pointB = new Coord2D(9, 8);
        grid.getTile(pointA).setType(Tile.TileType.TRAVERSABLE);
        grid.getTile(pointB).setType(Tile.TileType.TRAVERSABLE);
        
        System.out.println("Grid with pointA = " + pointA);
        System.out.println("          pointB = " + pointB);
        System.out.println(grid);
        
        Path path = new Path(grid, pointA, pointB, 1);
        path.setPathType(Tile.TileType.TRAVERSABLE, false);
        
        System.out.println("Grid with best route:\n" + grid);
    }
    
    public static void testGenerateGameGrid(int numOfGrids) {
        
        assert numOfGrids >= 0 : "You tried to generate a negative number of grids";
        
        Coord2D gridDimensions = new Coord2D(50, 50);
//        GameGrid2D grid = new GameGrid2D(gridDimensions);
//        System.out.println("grid:\n" + grid);
        
        for (int i = 0; i < numOfGrids; i++) {
            
            // Wait between each generation,
            // because randomization depends on the clock
            try {
                TimeUnit.MILLISECONDS.sleep(19);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            
            GameGrid2D grid = new GameGrid2D(gridDimensions, 5, 7);
            System.out.println("Grid #" + i + '\n' + grid + "\n\n");
        }
    }
    
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		System.out.println("Hello world!");

//		testSetOfPoints();
		
//		testGrid();
		
//		testMarkRow();
		
//		testMarkRect();
		
//		testPaths();
		
//		testIterator();
		
//		testDijkstra();
		
		if (args.length == 0)
		    testGenerateGameGrid(Integer.MAX_VALUE);
		else
		    testGenerateGameGrid(Integer.parseInt(args[0]));
	}

}
