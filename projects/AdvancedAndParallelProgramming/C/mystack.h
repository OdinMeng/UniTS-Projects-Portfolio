// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    08.06.2026

// This is the file that handles stack operations
// As explained in lesson, static stack should be OK. ALso considering to convering to a dynamic one if needed (might do it as a bonus)
// THIS IS THE NEXT THING TO DO. I'M LEAVING TOKENIZATION AND I/O HANDLING LAST

#include "tensors.h"
#include <stdlib.h>
#include <stdbool.h>


#ifndef MAX_STACK_SIZE 
#define MAX_STACK_SIZE 16 // Size of a stack; usually 16 is reasonable, can be increased at will
#endif

#ifndef _MYSTACK
#define _MYSTACK

// A stack can contain either a pointer to a tensor or a pointer to an array of characters. Either way, it contains pointers...

typedef enum {
	TYPE_TENSOR, TYPE_STRING
} StackDataType;

typedef struct {
	StackDataType type;
	union { tensor t; char * str; } value; // NOTE: value must be allocated with malloc
	int string_length; // Only if we have a string
}* StackElement;

typedef struct _myStack
{
	StackElement* stack;	
	int top;
	int stack_size;
}* myStack;

// Part 1: Generally useful stack operators, like push, pop, check if empty or full et cetera. DONE (TO CHECK)
myStack initializeStack(void);
bool isStackEmpty(myStack stack);
bool isStackFull(myStack stack);
int pushStack(myStack stack, StackElement elem);
StackElement popStack(myStack stack);
int expandStack(myStack stack);
int shrinkStack(myStack stack);

// Part 2: Specific stack manipulations. DONE (TO CHECK)
int my_dup(myStack stack);
int my_swap(myStack stack);
int my_over(myStack stack);
int my_drop(myStack stack);

// part 3: useful memory management stuff
void dropStack(myStack stack); // drop the entire stack, with its associated elements

void printStack(myStack stack); // print the entire stack, for visualization purposes

#endif 
