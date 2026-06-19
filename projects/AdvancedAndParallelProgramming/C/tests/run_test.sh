gcc -fopenmp -c -g tensors.c
gcc -c -g test_ops.c
gcc -fopenmp -g -ggdb3 -o test_ops  test_ops.o tensors.o