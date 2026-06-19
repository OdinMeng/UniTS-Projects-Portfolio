#include "mystack.h"
#include "tensors.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "token_interpreter.h"

int main(void)
{
	myStack S = initializeStack();

	tensor t = malloc(sizeof(*t));
	t -> shape[0]=1; t->shape[1] = 3;
	t -> ndim = 2;
	t -> total_references = 0;
	t -> on_mmap = 0;


	t -> data = (float*)malloc(sizeof(float)*3);

	t -> data[0] = 1.1; t -> data[1] = 2.1; t->data[2] = 3.5;
	
	StackElement tensor_to_push = (StackElement)malloc(sizeof(*tensor_to_push));
	tensor_to_push->type = TYPE_TENSOR;
	tensor_to_push->value.t = t; 

	pushStack(S, tensor_to_push);

	my_dup(S);

	printStack(S);

	interpret_binary_tensors(S, sum_tensors);

	printStack(S);
	my_dup(S);
	my_dup(S);
	my_dup(S);
	printStack(S);

	interpret_void(S, print_tensor);
	printStack(S);

	interpret_binary_tensors(S, mul_tensors_pointwise);
	printStack(S);

	char* s = malloc(sizeof(char)*32);
	strcpy(s, "./examples/fake.pgm\0");
	StackElement to_push = (StackElement)malloc(sizeof(*to_push));
	to_push->type = TYPE_STRING;
	to_push->value.str = s;
	to_push->string_length = strlen(s);
	pushStack(S, to_push);
	printStack(S);

	interpret_save(S, 1);
	printStack(S);

	char* snew = malloc(sizeof(char)*32);
	strcpy(s, "./examples/fake.pgm\0");
	StackElement to_pushnew = (StackElement)malloc(sizeof(*to_pushnew));
	to_pushnew->type = TYPE_STRING;
	to_pushnew->value.str = snew;
	to_pushnew->string_length = strlen(snew);
	pushStack(S, to_pushnew);
	printStack(S);

	interpret_load(S, 1);
	printStack(S);

	dropStack(S);
}