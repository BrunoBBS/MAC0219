all:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -o main -lpthread -lm -fopenmp -lstdc++
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -o main_serial -lpthread -lstdc++
	gcc src/mat_gen.cpp src/matrix.cpp src/pthreads.cpp src/util.cpp -o mat_gen -lm -fopenmp -lstdc++
gdb: debug

debug:
	gcc src/main.cpp src/matrix.cpp src/openmp.cpp src/pthreads.cpp src/util.cpp -O0 -g -o main -lpthread -lm -fopenmp -lstdc++
	gcc src/mat_gen.cpp src/matrix.cpp src/util.cpp -O0 -g -o mat_gen -lm -fopenmp -lstdc++
