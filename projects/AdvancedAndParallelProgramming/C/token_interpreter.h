// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    18.05.2026

// This file basically will "interpret" the tokens. Although the name is kind of misleading since it does NOT Make
// use of the token objects at all, the main idea is to provide an interface between the stack and the tensor operators

#include "tensors.h"
#include "mystack.h"
#include "io_tensors.h"
#include <stdlib.h>

typedef tensor (* binary_op_tensors)(tensor, tensor);  // Defines a generic operation between two tensors that returns a tensor
typedef tensor (* unary_op_tensor)(tensor);  // Defines a generic operation of one tensor which returns a tensor
typedef void (* unary_op_tensor_void)(tensor);  // Defines a generic operation of one tensor which does not do anything at all

// Each interpret_... function has the following return values:
// 0: Worked
// -1: Not worked (terminate the program immediately)

int interpret_binary_tensors(myStack S, binary_op_tensors op); // Interprets operations of type ( b a -- f(a,b) )
int interpret_dollar(myStack S); // Interprets b a m -- m?a:b specifically
int interpret_unary_tensor(myStack S, unary_op_tensor op); // Interpret operations of type ( a -- f(a) ) )
int interpret_void(myStack S, unary_op_tensor_void op); // INterpret operations of type ( a -- )
int interpret_save(myStack S, int PGM_flag); // Interprets ( filename -- tensor ). PGM or file with a flag; 1 if PGM, 0 if file 
int interpret_load(myStack S, int PGM_flag); // Interprets ( t filename -- )
