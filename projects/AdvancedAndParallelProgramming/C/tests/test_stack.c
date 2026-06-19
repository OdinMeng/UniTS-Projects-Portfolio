#include "mystack.h"
#include "tensors.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
	myStack S = initializeStack();
	printf("%d\n", isStackEmpty(S)); // expect 1
	char* s = malloc(sizeof(char)*15);
	strcpy(s, "hello, world");
	StackElement to_push = (StackElement)malloc(sizeof(*to_push));
	to_push->type = TYPE_STRING;
	to_push->value.str = s;
	to_push->string_length = 12;

	tensor t = malloc(sizeof(*t));
	t -> shape[0]=1; t->shape[1] = 3;
	t -> ndim = 2;
	t -> total_references = 0;
	t -> on_mmap = 0;


	t -> data = (float*)malloc(sizeof(float)*3);

	t -> data[0] = 1.1; t -> data[1] = 2.1; t->data[2] = 3.5;
	
	print_tensor(t);

	StackElement tensor_to_push = (StackElement)malloc(sizeof(*to_push));
	tensor_to_push->type = TYPE_TENSOR;
	tensor_to_push->value.t = t; 

	pushStack(S, tensor_to_push);
	pushStack(S, to_push);

	my_over(S);
	my_dup(S);

	dropStack(S);

}