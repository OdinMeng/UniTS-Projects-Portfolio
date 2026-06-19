gcc -c -g io_tensors.c
gcc -c -g tensors.c
gcc -c -g test_io.c
gcc -g -fopenmp -ggdb3 -o test_io test_io.o tensors.o io_tensors.o