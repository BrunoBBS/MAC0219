all:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp -o main -lpthread -lstdc++

gdb: debug

debug:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp -O0 -g -o main -lpthread -lstdc++
