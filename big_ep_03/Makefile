all:
	nvcc src/main.cu src/util.cpp src/cpu.cpp src/acc.cu -Iinclude -o ep3.mp3 -Xcompiler -fopenmp

clean:
	rm *.o
	rm ep3.mp3