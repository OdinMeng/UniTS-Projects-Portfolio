// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    19.05.2026

// This file basically will "interpret" the tokens. Although the name is kind of misleading since it does NOT Make
// use of the token objects at all, the main idea is to provide an interface between the stack and the tensor operators

#include "tensors.h"
#include "mystack.h"
#include "io_tensors.h"
#include <stdlib.h>
#include <stdio.h>
#include "token_interpreter.h"

int interpret_binary_tensors(myStack S, binary_op_tensors op)
{
	//	Inteprets a generic binary operator on tensors which affects the stack as ( a b -- op(a,b) )
	//	Inputs:
	//		myStack S: The stack in question.
	//		binary_op_tensors op: A binary operator tensor which returns a tensor, so something like op: T x T -> T
	//	Outputs: An integer indicating failure (-1) or success (0). Will be the same for all functions of this file

	// Step 1: check if stack is big enough or not
	if(S->top < 1){ printf("ERRROR: Stack is not big enough for a binary operation. Terminating execution.\n"); return -1;}

	// Step 1.1. Pop two items and check that they're both tensors
	StackElement e1 = popStack(S);
	StackElement e2 = popStack(S);

	if( (e1->type != TYPE_TENSOR) || (e2->type != TYPE_TENSOR) )
	{
		printf("ERROR: Invalid stack element types. Terminating execution.\n");
		// revert pop by pushing stuff back
		pushStack(S, e2); 
		pushStack(S, e1); 

		return -1;
	}

	// Step 2. Execute operation and check their return value (if NULL, some error occurred)
	tensor t1 = e1->value.t;
	tensor t2 = e2->value.t;

	tensor t3 = op(t1, t2);

	// free t1, t2 if nothing is left; check if they're equal! (possible with duplications or overs);
	if(t1 != t2)
	{
		if(t1->total_references == 0) free_tensor(t1); 
		if(t2->total_references == 0) free_tensor(t2);
	}
	else
	{
		if(t1->total_references == 0) free_tensor(t1);
	}

	free(e1);
	free(e2);


	if(t3 == NULL)
	{
		printf("Error occurred in operation. Terminating execution.\n"); 

		return -1;
	}

	// Step 3. If all OK, push the new tensor to stack and return 0;
	else
	{
		t3->total_references = 0;
		StackElement e3 = malloc(sizeof(*e3)); 
		e3->type = TYPE_TENSOR;
		e3->value.t = t3;

		if(pushStack(S, e3) == 0)
		{
			return 0;
		}
		else
		{
			printf("Error in pushing element. Terminating execution.\n");
			free(e3);
			free_tensor(t3);
			return -1;

		}
	}
}

int interpret_dollar(myStack S)
{
	//	Interprets the operator ( b a m -- m?a:b ) specifically
	//	Inputs:
	//		myStack S: The stack in question.
	//	Outputs: An integer indicating failure (-1) or success (0). Will be the same for all functions of this file

	// KInda same as before but I have three tensors

	// Step 1: check if stack is big enough or not
	if(S->top < 2){ printf("ERRROR: Stack is not big enough for $ operation. Terminating execution.\n"); return -1;}

	// Step 1.1. Pop two items and check that they're both tensors
	StackElement e1 = popStack(S);
	StackElement e2 = popStack(S);
	StackElement e3 = popStack(S);

	if( (e1->type != TYPE_TENSOR) || (e2->type != TYPE_TENSOR) || (e3->type != TYPE_TENSOR) )
	{
		printf("ERROR: Invalid stack element types. Terminating execution.\n"); 

		pushStack(S, e3);
		pushStack(S, e2);
		pushStack(S, e1);

		return -1;
	}

	// Step 2: Execute operation and check for soundness
	tensor m = e1->value.t;
	tensor a = e2->value.t;
	tensor b = e3->value.t;

	tensor t_res = select_a_or_b(m, a, b);


	free(e1);
	free(e2);
	free(e3);

	// Before freeing tensors (discarded), check if they're equal and ofc check for total references
	if( (a == b) && (a == m)) // all equals by transitive property
	{
		if(a->total_references == 0) free_tensor(a);
	}
	else if ( a == m ) // only a == m
	{
		if(a->total_references == 0) free_tensor(a);
		if(b->total_references == 0) free_tensor(b);
	}
	else if ( b == m ) // only b == m
	{
		if(a->total_references == 0) free_tensor(a);
		if(b->total_references == 0) free_tensor(b);
	}
	else if ( a == b ) // only a == b 
	{
		if(m->total_references == 0) free_tensor(m); 
		if(a->total_references == 0) free_tensor(a);
	}
	else // all different
	{
		if(m->total_references == 0) free_tensor(m); 
		if(a->total_references == 0) free_tensor(a);
		if(b->total_references == 0) free_tensor(b);
	}

	if(t_res == NULL)
	{
		printf("Error occurred in operation $. Terminating\n");
		return -1;
	}

	// Step 3. attempt push
	else
	{
		t_res->total_references = 0;
		StackElement e_res = malloc(sizeof(*e_res)); 
		e_res->type = TYPE_TENSOR;
		e_res->value.t = t_res;

		if(pushStack(S, e_res) == 0)
		{
			return 0;
		}
		else
		{
			printf("Error in pushing element. Terminating execution.\n");
			free(e_res);
			free_tensor(t_res);
			return -1;

		}
	}

}

int interpret_unary_tensor(myStack S, unary_op_tensor op) 
{
	//	Inteprets a generic unary operator on tensors which affects the stack as ( a -- op(a) )
	//	Inputs:
	//		myStack S: The stack in question.
	//		unary_op_tensor op: A unary operator tensor which returns a tensor, so something like op: T -> T
	//	Outputs: An integer indicating failure (-1) or success (0). Will be the same for all functions of this file

	// Step 1: check if stack is big enough or not
	if(S->top < 0){ printf("ERRROR: Stack is not big enough for a binary operation. Terminating execution.\n"); return -1;}

	// Step 1.1. Pop two items and check that they're both tensors
	StackElement e1 = popStack(S);

	if( (e1->type != TYPE_TENSOR) )
	{
		printf("ERROR: Invalid stack element types. Terminating execution.\n"); 
		pushStack(S, e1);
		return -1;
	}

	// Step 2. Execute operation and check their return value (if NULL, some error occurred)
	tensor t1 = e1->value.t;
	tensor t2 = op(t1);

	// free t1, t2 if nothing is left
	if(t1->total_references == 0) free_tensor(t1); 

	free(e1);

	if(t2 == NULL)
	{
		printf("Error occurred in operation. Terminating execution.\n"); 

		return -1;
	}

	// Step 3. If all OK, push the new tensor to stack and return 0;
	else
	{
		t2->total_references = 0;
		StackElement e2 = malloc(sizeof(*e2)); 
		e2->type = TYPE_TENSOR;
		e2->value.t = t2;

		if(pushStack(S, e2) == 0)
		{
			return 0;
		}
		else
		{
			printf("Error in pushing element. Terminating execution.\n");
			free(e2);
			free_tensor(t2);
			return -1;

		}
	}
}

int interpret_void(myStack S, unary_op_tensor_void op)
{
	//	Inteprets a generic unary operator on tensors which affects the stack as ( a -- )
	//	Inputs:
	//		myStack S: The stack in question.
	//		unary_op_tensor op: A unary operator tensor which returns nothing, so something like op: T -> NONE
	//	Outputs: An integer indicating failure (-1) or success (0). Will be the same for all functions of this file.
	//	N.B. The intended use case is the print operator (p), however this frameworks allows for easy generalization

	// Step 1: check if stack is big enough or not
	if(S->top < 0){ printf("ERRROR: Stack is not big enough for a binary operation. Terminating execution.\n"); return -1;}

	// Step 1.1. Pop two items and check that they're both tensors
	StackElement e1 = popStack(S);

	if( (e1->type != TYPE_TENSOR) )
	{
		printf("ERROR: Invalid stack element types. Terminating execution.\n"); 
		pushStack(S, e1);
		return -1;
	}

	// Step 2. Execute operation and that's it!
	tensor t1 = e1->value.t;

	op(t1);

	// free t1, t2 if nothing is left
	if(t1->total_references == 0) free_tensor(t1); 
	free(e1);

	return 0;
}

int interpret_save(myStack S, int PGM_flag)
{
	//	Interprets a generic save operator, so an operator which has effect on the stack as ( t f -- )
	//	Inputs:
	//		myStack S: The stack in question.
	//		int PGM_flag: Specify whether to load it from a .pgm file or a from .bin file
	//	Outputs: An integer indicating failure (-1) or success (0). Will be the same for all functions of this file.

	if(S->top < 1){ printf("ERRROR: Stack is not big enough for a save operation. Terminating execution.\n"); return -1;}

	// Step 1.1. Pop two items and check that the first is a string another is a tensor
	StackElement e1 = popStack(S);
	StackElement e2 = popStack(S);

	if( (e1->type != TYPE_STRING) || (e2->type != TYPE_TENSOR))
	{ 
		printf("ERROR: Incompatible types for a save opration. Terminator.\n"); 
		pushStack(S, e2); pushStack(S, e1);
		return -1; 
	}

	char* filename = e1->value.str;
	tensor t = e2->value.t; 

	// Preliminary check: two elements and one is a tensor another is a strng
	if(PGM_flag == 1) write_pgm(t, filename);
	else write_tensor(t, filename);
	// free popped items (if needed)
	free(e1->value.str);
	if(e2->value.t->total_references == 0) free_tensor(e2->value.t);

	free(e1);
	free(e2);

	return 0;

}

int interpret_load(myStack S, int PGM_flag)
{
	//	Interprets a generic load operator, so an operator which has effect on the stack as ( f -- t )
	//	Inputs:
	//		myStack S: The stack in question.
	//		int PGM_flag: Specify whether to load it from a .pgm file or a from .bin file
	//	Outputs: An integer indicating failure (-1) or success (0). Will be the same for all functions of this file.

	if(S->top < 0){ printf("ERRROR: Stack is not big enough for a load operation. Terminating execution.\n"); return -1;}
	// Step 1.1. pop a string
	StackElement e = popStack(S);
	if(e->type != TYPE_STRING)
	{
		printf("ERROR: Incompatible type for a load operation. BÌjaiedfawji\n");
		pushStack(S, e);
		return -1;
	}

	char* filename = e->value.str;

	tensor t; 

	if(PGM_flag == 1) t = read_pgm(filename);
	else t  = read_file_mmap(filename);

	if(t == NULL)
	{
		printf("Load operation failed. Terminating.\n");
		pushStack(S, e);
		return -1;
	}

	else
	{
		t->total_references=0;

		StackElement to_push = malloc(sizeof(*to_push));

		to_push->type = TYPE_TENSOR;
		to_push->value.t = t;

		if(pushStack(S, to_push) != 0)
		{
			printf("Push failed.\n");
			free_tensor(t);
			free(to_push);
			pushStack(S, e);
			return -1;
		}
		else 
		{
			free(filename);
			free(e);
			return 0;
		}
	}
}