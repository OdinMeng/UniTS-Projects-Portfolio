gcc -c -g test_tokens.c
gcc -c -g tokenizer.c
gcc -g -fopenmp -ggdb3 -o test_tokens test_tokens.o tokenizer.o