gcc -c -g test_token_interpreter.c
gcc -c -g mystack.c
gcc -c -g token_interpreter.c
gcc -c -g tensors.c
gcc -c -g io_tensors.c
gcc -g -fopenmp -ggdb3 -o test_token_interpreter test_token_interpreter.o tokenizer.o mystack.o tensors.o io_tensors.o token_interpreter.o