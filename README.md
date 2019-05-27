Test environment for procedurally generating levels.

# Usage
If using for first time, do
```
chmod +x compile_and_run.sh
```

You need only complete the above step once. After that, use ```./compile_and_run.sh java``` or ```./compile_and_run.sh c++``` to compile and run the project.

Currently, this program generates 2,147,483,647 different playable grids and prints them all to the terminal. (This is the maximum value of an integer in Java.) You can optionally use ```./compile_and_run.sh java 5``` or ```./compile_and_run.sh c++ 5``` to generate only 5 grids instead (or however many grids you'd like to generate).

Please note that the commands ```java``` and ```javac``` are required. You may need to install these if you do not already have them.
