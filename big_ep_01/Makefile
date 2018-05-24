all:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -o main -lpthread -fopenmp -lstdc++
	gcc src/mat_gen.cpp src/matrix.cpp src/pthreads.cpp src/util.cpp -o mat_gen -fopenmp -lstdc++
gdb: debug

debug:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -O0 -g -o main -lpthread -fopenmp -lstdc++
	gcc src/mat_gen.cpp src/matrix.cpp src/util.cpp -O0 -g -o mat_gen -fopenmp -lstdc++
