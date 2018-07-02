all:configure
	mpicxx src/main.cpp -Iinclude -fopenmp -c -o build/main.o -g
	g++ src/util.cpp -Iinclude -c -o build/util.o -g
	g++ src/cpu.cpp -Iinclude -c -o build/cpu.o -g -fopenmp
	nvcc src/acc.cu -Iinclude -c -o build/acc.o -g
	mpicxx build/main.o build/util.o build/cpu.o build/acc.o -o ep3 -L/usr/local/cuda/lib64 -lcudart  -fopenmp

configure:
	mkdir -p build

clean:
	rm -r build
	rm ep3