// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    19.05.2026

// This is the header file that defines the tokens structure and the associated constants

#include <stdlib.h>
#include <stdio.h>
#include "tensors.h"

#ifndef _CONSTANTS_TOKENIZER

#define MAX_TOKENARRAY_SIZE 1024 // define maximum amount of elements in a token array
#define INITIAL_FLOAT_BUFSIZE 32 // Initial float buffer size for tensor parsing
#define INITIAL_STRF_BUFSIZE 16 // Initial char* buffer size for string (or float) parsing, can be doubled at will
// Maximum buffer for a possible token (reasonable, only "longest" ones can be strings)
// while flots should have at most 6-7 digits due to fp32 precision
// strings should only represent directories to file, so they shouldn't be TOO long (idk what sort of system has filepaths longer than 64 characters...)

#endif

#ifndef _TOKENS
#define _TOKENS


typedef enum {
	TOKEN_EOF = 0,
	TOKEN_NUMBER, // Either float or integer,
	TOKEN_STRING,
	TOKEN_OPERATOR, // Operators on StackElement(s)
	TOKEN_LBRACKET, TOKEN_RBRACKET, // Represents either [ or ],
} TokenType;

typedef enum {
	OP_PLUS = 0,
	OP_MINUS,
	OP_PROD,
	OP_CMP_LESS,
	OP_CMP_MORE,
	OP_CMP_EQ,
	OP_AND,
	OP_OR,
	OP_NOT,
	OP_DOLLAR,
	OP_MATRIX_PROD,
	OP_DOT_PROD,
	OP_CONV,
	OP_RESHAPE,
	OP_FLATTEN,
	OP_SHAPE,
	OP_RAND,
	OP_RELU, 
	OP_MIN,
	OP_MAX,
	OP_REDUCT_SUM,
	OP_FILL,
	OP_PRINT,
	OP_DUPLICATE,
	OP_SWAP,
	OP_OVER,
	OP_DROP,
	OP_READPGM,
	OP_WRITEPGM,
	OP_READFILE,
	OP_WRITEFILE,
} OperatorType;

typedef struct 
{
	TokenType token;
	union {
		float numerical_value;
		OperatorType operator_type;
		char* string;
	} value; 
} Token;

typedef struct 
{
	FILE* fp;
	Token t_curr;
	int is_loaded;
	
}* TokenStream;

TokenStream tokenstream_open(char* filename);
void tokenstream_close(TokenStream ts);
int tokenstream_next(TokenStream ts, Token* token_ptr); // Advances file reading and loads the next token into token_ptr
tensor parse_tensor(TokenStream ts); 

#endif