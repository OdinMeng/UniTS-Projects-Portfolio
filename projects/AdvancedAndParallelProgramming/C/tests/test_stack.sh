gcc -c -g test_stack.c
gcc -c -g tensors.c
gcc -g -fopenmp -ggdb3 -o test_stack test_stack.o tensors.o