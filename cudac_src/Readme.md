# Grid Generation
### Overview and Usage
This code will generate an MxN grid and populate it with a random path from the lower corner to the upper corner. When reading the output, t indicates a traversable tile, whereas N denotes an obstacle. There are currently 3 different implementations of this code; One for Java, one for C++ and one for Cuda using C++. In order to use this program, first compile it using the provided Makefile. Currently, a Makefile exists for the C++ implementation and for the Cuda implementation. The Makefile will generate an executable named procgen. Run procgen with the command ./procgen or with ./procgen n, (where n is any positive integer). Running the program with no arguments will generate a multitude of grids one after the other. Passing in an argument determines how many grids will be created. 

Currently, there is no way to control the size of the grid by passing in arguments. The default setting for grid generation is a 50x50 grid, with a path width of 5 and 7 landmarks. Should you wish to change anything about the grid or the generated paths, you must open MainDriver.cpp. On line 22 you may adjust the grid size, on line 32 you can adjust width and the number of landmarks, (see the comments in code for more details).

### C++ Version
The C++ version uses Dijkstra\'s algorithm to find the shortest path between two landmarks. The shortest path then becomes the path used by the grid.

### Cuda Version
This version sought to improve the performance of the C++ implementation. However, due to complications with parallelizing Dijkstra\'s algorithm, a new method was developed for constructing the grid. This method assigns one thread for each path that needs to be generated. Each thread begins at a specific coordinate, and steps along the grid until they reach their destination.

NOTE: You must be using an Nvidia graphics card capable of supporting sm_35 in order to run this version.

### Known Issues
We have not yet tested for memory leaks with the Cuda implementation. It is advised that you do not use ./procgen without any arguments as hidden memory leaks could lead to your system crashing. 

Since the C++ implementation does not use a similar algorithm to the Cuda implementation, we are not yet sure if there is any tangible benefit to using parallel processing over a simple serilaized approach.  
