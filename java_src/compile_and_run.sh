(echo "Compiling...") \
&& (javac *.java) \
&& (echo "Compiled successfully! Running...") \
&& (echo "") \
&& (java -ea -classpath . MainDriver $1)

