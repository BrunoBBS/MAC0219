all:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -o main -lpthread -lstdc++

gdb: debug

debug:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -O0 -g -o main -lpthread -lstdc++
