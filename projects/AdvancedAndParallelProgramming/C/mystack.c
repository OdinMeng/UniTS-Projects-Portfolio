// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    08.06.2026

// This file implements the operations on stacks

#include "mystack.h"
#include <stdlib.h>
#include <stdbool.h>
#include "tensors.h"
#include <stdio.h>
#include <string.h>

myStack initializeStack(void)
{
	// Creates an empty stack. Self explanatory, no documentation needed for this one
	myStack stack = (myStack)malloc(sizeof(*stack));
	stack->top = -1;
	stack->stack_size = MAX_STACK_SIZE;
	stack->stack = malloc(sizeof(StackElement)*MAX_STACK_SIZE);
	return stack;
}

int expandStack(myStack stack)
{
	// Expands stack by the double of its current size. Uses realloc to do it. If it fails, return -1
	// If OK, returns 0

	// Requires only the stack as the argument

	StackElement* new_stack = realloc(stack->stack, sizeof(StackElement)*((stack->stack_size) * 2));
	if(new_stack == NULL)
	{
		perror("realloc failed:");
		return -1;
	}

	else
	{
		stack->stack = new_stack;
		stack->stack_size *= 2;
		return 0;
	}
}

int shrinkStack(myStack stack)
{
	// Shrinks stack by halving its size. 
	// Returns integer: 0 if success, -1 if failure
	// Accepts stack as argument

	// Checks if the operation can be done in the first place
	if(stack->top >= stack->stack_size / 2)
	{
		printf("ERROR: Cannot shrink stack in the current state");
		return -1;
	}

	StackElement* new_stack = realloc(stack->stack, sizeof(StackElement)*((stack->stack_size) / 2));
	if(new_stack == NULL)
	{
		perror("realloc failed:");
		return -1;
	}

	else
	{
		stack->stack = new_stack;
		stack->stack_size /= 2;
		return 0;
	}
}

bool isStackEmpty(myStack stack)
{
	// Checks if the stack is empty by just checking the top index
	return (bool)(stack->top == -1);
}

bool isStackFull(myStack stack)
{
	// Checks if the stack is full by checking the top index
	return (bool)(stack->top+1 >= stack->stack_size);
}

int pushStack(myStack stack, StackElement elem)
{
	//	Pushes an element.
	//	Input(s):
	//		myStack stack: the stack
	//		StackElement elem: the StackElement element which we want to push the stack into
	//	Ouput: Returns 0 on success, -1 on failure


	if(isStackFull(stack))
	{
		/*
		printf("ERROR: Failed Stack Push due to Stack overflow.\n");
		stack->top--;
		*/
		//printf(">> WARNING: STACK OVERFLOW. ATTEMPING TO EXPAND STACK.\n");
		if (expandStack(stack) != 0)
		{
			printf("ERROR: Stack expansion failed.\n");
			return -1;
		}
	}

	stack->top++;
	stack->stack[stack->top] = elem;

	// increase tensor refs. if it's a tensor
	if(elem->type == TYPE_TENSOR) (elem->value).t->total_references++;
	return 0;
}

StackElement popStack(myStack stack)
{
	//	Pops the stack by selecting the top-most element.
	//	Input(s):
	//		myStack stack: the stack
	//	Ouput: Returns the resulting StackElement

	// Pops an element from the stack
	if(isStackEmpty(stack))
	{
		return NULL; // ERROR: EMPTY STACK
	}
	StackElement popped = stack->stack[stack->top];

	// If tensor decrease its total references
	if(popped->type == TYPE_TENSOR) (popped->value).t->total_references--;

	stack->top--;

	// Shrink stack if it's "appropriate enough" 
	// (current top is 1/4th of the current stack size and the current size exceeds MAX_STACK_SIZE)

	if ( (stack->top+1) <= stack->stack_size/4 && stack->stack_size > MAX_STACK_SIZE)
	{
		int shrink_result = shrinkStack(stack);
		if(shrink_result != 0)
		{
			printf("ERROR: Shrink stack failed");
			free(popped);
			return NULL;
		}
	} 

	return popped;
}

int aux_dup_over(myStack stack, StackElement elem)
{
	// 	Copies an element and pushes it into the stack (useful for duplication or over operation)
	//	Input(s):
	//		myStack stack: the stack
	//		StackElement elem: the element in question
	//	Ouput: Returns a number which tells success or failure

	// create new StackElement and copy stuff

	StackElement new_elem = malloc(sizeof(*new_elem));
	new_elem->type = elem->type;
	new_elem->value = elem->value;
	new_elem->string_length = elem->string_length;

	// If tensor just copy the pointer 
	if(elem->type == TYPE_TENSOR)
	{
		new_elem->value.t = elem->value.t;
	}

	// If string copy the contents entirely
	if(elem->type == TYPE_STRING)
	{
		new_elem->value.str = malloc(sizeof(char)*(strlen(elem->value.str)+1));
		strcpy(new_elem->value.str, elem->value.str);
	}

	int push_result = pushStack(stack, new_elem);

	if(push_result != 0)
	{
		if(elem->type == TYPE_STRING)
			free(new_elem->value.str);
		free(new_elem);
	}

	return push_result;
}

int my_dup(myStack stack)
{
	// 	Implements duplication operator
	//	Input(s):
	//		myStack stack: the stack
	// 	Returns -1 if error, 0 if OK

	if(isStackEmpty(stack))
	{
		return -1; // ERROR: EMPTY STACK
	}
	StackElement last_elem = stack->stack[stack->top];
	return aux_dup_over(stack, last_elem);
}

int my_swap(myStack stack)
{
	// 	Implements swap operator
	//	Input(s):
	//		myStack stack: the stack
	// 	Returns -1 if error, 0 if OK

	// This time I must specifically check for almost two elements
	if(stack->top<1)
	{
		return -1; // STACK UNDERFLOW
	}
	StackElement a = stack->stack[stack->top];
	StackElement b = stack->stack[stack->top-1];

	stack->stack[stack->top] = b;
	stack->stack[stack->top-1] = a;
	return 0; // OK
}

int my_over(myStack stack)
{
	// 	Implements over operator
	//	Input(s):
	//		myStack stack: the stack
	// 	Returns -1 if error, 0 if OK

	// This time I must specifically check for almost two elements
	if(stack->top<1)
	{
		return -1; // STACK OVERFLOW
	}
	StackElement b = stack->stack[stack->top-1];
	return aux_dup_over(stack, b);
}

int my_drop(myStack stack)
{
	// 	Implements drop operator
	//	Input(s):
	//		myStack stack: the stack
	// 	Returns -1 if error, 0 if OK

	if(isStackEmpty(stack))
	{
		printf("ERROR: Attempted to drop in an empty stack");
		return -1; // ERROR: EMPTY STACK
	}
	int old_top = stack->top;
	StackElement popped = popStack(stack); // pop the stack and do nothing

	// if it the was the last element: free the element
	if(old_top != stack->top)
	{
		if(popped->type == TYPE_STRING)
		{
			free(popped->value.str);
		}
		else if(popped->type == TYPE_TENSOR)
		{
			// Must also account for total references (not only relative to StackElement references to it)
			tensor t = popped->value.t; 

			if(t->total_references <= 0) free_tensor(t);
		}
		free(popped);
	}
	return 0;
}

void dropStack(myStack stack)
{
	// 	Drops the entire stack. Useful for program termination (either error or EOF token)
	//	Input: myStack stack, the stack
	//	Ouput: None, void

	if(isStackEmpty(stack))
	{
		free(stack->stack);
		free(stack);
		return; // already empty: just free the pointer pointing to the stack
	}
	else
	{
		while(stack->top >= 0) my_drop(stack);
	}

	free(stack->stack);
	free(stack);
}

void printStack(myStack stack)
{
	//	Useful debugging operator: prints the entire stack, by iterating over it,
	//	with the associated string for strings or the shape and ndim for tensors.
	//	Input: myStack stack, the stack
	//	Ouput: None, prints stuff
	printf("====== PRINTING STACK ======\n");
	printf("STACK SIZE: %d\n", stack->stack_size);
	printf("STACK TOP: %d\n", stack->top);

	if(stack->top == -1)
	{
		printf("EMPTY STACK. FINISHED\n");
	}

	else
	{
		for(int i=0; i<=stack->top; i++)
		{
			StackElement elem = stack->stack[i];
			if(elem->type == TYPE_TENSOR)
			{
				printf("%d: Tensor with %d references\n", i, elem->value.t->total_references);
				printf("\tndims: %d\n", elem->value.t->ndim);

				for(int i=0; i<elem->value.t->ndim; i++)
				{
					printf("\tshape[%d]: %d\n", i, elem->value.t->shape[i]);
				}
				// printf("\t");
				// print_tensor(elem->value.t);
			}

			else if(elem->type == TYPE_STRING)
			{
				printf("%d: String\n\t%s\n", i, elem->value.str);
			}
		}
	}
	printf("====== STACK OVER ======\n");
}