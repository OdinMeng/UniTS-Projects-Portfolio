// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    27.05.2026

// This is the "entry point code", which will read the a TensorForth code and handle the stack as per the project guidelines.
// Things to be careful of: ERROR MANAGEMENT

#include <stdio.h>
#include "tokenizer.h"
#include "tensors.h"
#include "mystack.h"
#include "io_tensors.h"
#include "token_interpreter.h"
#include <stdbool.h>

#define VERBOSE 0 // set 0 to final delivery

int main(int argc, char* argv[])
{
    //  The main file, which does everything from tokenizing to executing tensor operations.
    //  Inputs:
    //      argc: Number of arguments. Must be 2, first one being program name (tensorforth) and second one being filename of the
    //              file we want to interpret
    //      argv: Array of strings, we are interested in argv[1]
    //  Ouputs: Integer value, 0 for success -1 for wrong usage of the program, 1 for insuccess during runtime
    
    int retval = 0;

	if(argc != 2) // MUST have two arguments: 1st is program name, 2nd is filename
	{
		printf("TensorForth\n\tDeveloped by Dino Meng SM3201466\n\tTo use this program please use it as the following: 'tensorforth <program_name>'\n");
	    return -1;
    }

	// Step 0: Initialize stack
	myStack S = initializeStack();

	// Step 1: Tokenize the script (get a TokenStream)
    TokenStream ts = tokenstream_open(argv[1]);

    if (ts == NULL) {
    	free(S);
        return 1;
    }

    Token token_cursor;

    // Main loop: Keep extracting tokens from the tokenstream and interpret them
    while (tokenstream_next(ts, &token_cursor) == 0)
    {
        if (token_cursor.token == TOKEN_EOF)
        {
            retval = 0;
            break;
        }

        if (token_cursor.token == TOKEN_NUMBER)
        {
            if(VERBOSE == 1) printf("Processing a number...\n");

            printf("Error parsing tensor: foreign number (must be contained in a tensor). Terminating\n");
            retval = 1;
            break;
        }

        if (token_cursor.token == TOKEN_STRING)
        {
        	// push the string into stack
            if(VERBOSE == 1) printf("Processing a string...\n");

            // Step 1: create string StackElement
            StackElement to_push_string = malloc(sizeof(*to_push_string));

            to_push_string->type = TYPE_STRING;
            to_push_string->value.str = malloc(sizeof(char)*(strlen(token_cursor.value.string)+1)); // +1 to account for '\0'
            strcpy(to_push_string->value.str, token_cursor.value.string);

            if(pushStack(S, to_push_string) != 0)
            {
                printf("Failed Stack Push.\n");
                free(to_push_string->value.str);
                free(to_push_string);
                dropStack(S);
                retval = 1;
                break;
            }

            // remember to free the string buffer of the token
            free(token_cursor.value.string);
        }

        if (token_cursor.token == TOKEN_OPERATOR)
        {
            if(VERBOSE == 1) printf("Processing an operator... ");
        	// do the operator and check for errors
            int operator_result; 

            if(token_cursor.value.operator_type == OP_PLUS)
            {
                if(VERBOSE == 1) printf("+");
                operator_result=interpret_binary_tensors(S, sum_tensors);
            }
            else if(token_cursor.value.operator_type == OP_MINUS)
            {
                if(VERBOSE==1) printf("-");
                operator_result=interpret_binary_tensors(S, diff_tensors);
            }
            else if(token_cursor.value.operator_type == OP_PROD)
            {
                if(VERBOSE==1) printf("*");
                operator_result=interpret_binary_tensors(S, mul_tensors_pointwise);

            }
            else if(token_cursor.value.operator_type == OP_CMP_LESS)
            {
                if(VERBOSE==1) printf("<");
                operator_result=interpret_binary_tensors(S, cmp_less_tensors);
            }
            else if(token_cursor.value.operator_type == OP_CMP_MORE)
            {
                if(VERBOSE==1) printf(">");
                operator_result=interpret_binary_tensors(S, cmp_less_tensors);

            }
            else if(token_cursor.value.operator_type == OP_CMP_EQ)
            {
                if(VERBOSE==1) printf("=");
                operator_result=interpret_binary_tensors(S, cmp_eq_tensors);

            }
            else if(token_cursor.value.operator_type == OP_AND)
            {
                if(VERBOSE==1) printf("&");
                operator_result=interpret_binary_tensors(S, and_tensors);
            }
            else if(token_cursor.value.operator_type == OP_OR)
            {
                if(VERBOSE==1) printf("|");
                operator_result=interpret_binary_tensors(S, or_tensors);
            }
            else if(token_cursor.value.operator_type == OP_NOT)
            {
                if(VERBOSE==1) printf("!");
                operator_result=interpret_unary_tensor(S, not_tensor);
            }
            else if(token_cursor.value.operator_type == OP_DOLLAR)
            {
                if(VERBOSE==1) printf("$");
                operator_result=interpret_dollar(S);
            }
            else if(token_cursor.value.operator_type == OP_MATRIX_PROD)
            {
                if(VERBOSE==1) printf("@");
                operator_result=interpret_binary_tensors(S, product_tensors);
            }
            else if(token_cursor.value.operator_type == OP_DOT_PROD)
            {
                if(VERBOSE==1) printf(".");
                operator_result=interpret_binary_tensors(S, dot_product_tensors);

            }
            else if(token_cursor.value.operator_type == OP_CONV)
            {
                if(VERBOSE==1) printf("c");
                operator_result=interpret_binary_tensors(S, conv_tensors);
            }
            else if(token_cursor.value.operator_type == OP_RESHAPE)
            {
                if(VERBOSE==1) printf("r");
                operator_result=interpret_binary_tensors(S, reshape_tensor);
            }
            else if(token_cursor.value.operator_type == OP_FLATTEN)
            {
                if(VERBOSE==1) printf("_");
                operator_result=interpret_unary_tensor(S, flatten_tensor);
            }
            else if(token_cursor.value.operator_type == OP_SHAPE)
            {
                if(VERBOSE==1) printf("#");
                operator_result=interpret_unary_tensor(S, get_tensor_shape);
            }
            else if(token_cursor.value.operator_type == OP_RAND)
            {
                if(VERBOSE==1) printf("?");
                operator_result=interpret_unary_tensor(S, rand_tensor);

            }
            else if(token_cursor.value.operator_type == OP_RELU)
            {
                if(VERBOSE==1) printf("ReLU");
                operator_result=interpret_unary_tensor(S, relu_tensor);

            }
            else if(token_cursor.value.operator_type == OP_MIN)
            {
                if(VERBOSE==1) printf("m");
                operator_result=interpret_binary_tensors(S, min_tensor);
            }

            else if(token_cursor.value.operator_type == OP_MAX)
            {
                if(VERBOSE==1) printf("M");
                operator_result=interpret_binary_tensors(S, max_tensor);
            }
            
            else if(token_cursor.value.operator_type == OP_REDUCT_SUM)
            {
                if(VERBOSE==1) printf("Sigma");
                operator_result=interpret_unary_tensor(S, reduct_sum_tensor);
            }
            
            else if(token_cursor.value.operator_type == OP_FILL)
            {
                if(VERBOSE==1) printf("f");
                operator_result=interpret_binary_tensors(S, fill_shape_value);
            }
            
            else if(token_cursor.value.operator_type == OP_PRINT)
            {
                if(VERBOSE==1) printf("p");
                operator_result=interpret_void(S, print_tensor);
            }
            
            else if(token_cursor.value.operator_type == OP_DUPLICATE)
            {
                if(VERBOSE==1) printf("d");
                operator_result=my_dup(S);
            }
            
            else if(token_cursor.value.operator_type == OP_SWAP)
            {
                if(VERBOSE==1) printf("s");
                operator_result=my_swap(S);
            }
            
            else if(token_cursor.value.operator_type == OP_OVER)
            {
                if(VERBOSE==1) printf("o");
                operator_result=my_over(S);
            }
            
            else if(token_cursor.value.operator_type == OP_DROP)
            {
                if(VERBOSE==1) printf("D");
                operator_result=my_drop(S);

            }
            
            else if(token_cursor.value.operator_type == OP_READPGM)
            {
                if(VERBOSE==1) printf("(");
                operator_result=interpret_load(S, 1);
            }
            
            else if(token_cursor.value.operator_type == OP_WRITEPGM)
            {
                if(VERBOSE==1) printf(")");
                operator_result=interpret_save(S, 1);
            }
            
            else if(token_cursor.value.operator_type == OP_READFILE)
            {
                if(VERBOSE==1) printf("{");
                operator_result=interpret_load(S, 0);
            }
            
            else if(token_cursor.value.operator_type == OP_WRITEFILE)
            {
                if(VERBOSE==1) printf("}");
                operator_result=interpret_save(S, 0);
            }
            else
            {
                if(VERBOSE==1) printf("unknown");
                printf("ERROR: Illegal operator detected. Terminating program.\n");
                retval = 1;
                break;
            }

            if(VERBOSE==1)printf("\n");

            if(operator_result != 0)
            {
                printf("ERROR: Operation failed. Retreat\n");
                retval = 1;
                break;
            }
        }

        if (token_cursor.token == TOKEN_LBRACKET)
        {
            if(VERBOSE == 1) printf("Processing a tensor...\n");
        	// start reading tensor: hardest part!
        	tensor t_parsed = parse_tensor(ts);

        	if(t_parsed == NULL)
        	{
        		printf("Error parsing tokens: Unrecognized tensor. Terminating.\n");
                retval = 1;
                break;
        	}

            if(VERBOSE == 1)print_tensor(t_parsed);

        	StackElement to_push = malloc(sizeof(*to_push));
        	to_push->type = TYPE_TENSOR;
        	to_push->value.t = t_parsed;
        	t_parsed -> total_references = 0;

        	if(pushStack(S, to_push) != 0)
            {
                printf("sTACK push failed\n");
                retval = 1;
                break;
            }
        }

        if (token_cursor.token == TOKEN_RBRACKET)
        {
            if(VERBOSE == 1) printf("Processing a ]...\n");
        	printf("Error parsing tensor: foreign ] symbol. Terminating\n");
            retval = 1;
            break;
        }

        if(VERBOSE == 1) printStack(S);
    }

    // free stuff
    dropStack(S);
    tokenstream_close(ts);

    return retval;
}

