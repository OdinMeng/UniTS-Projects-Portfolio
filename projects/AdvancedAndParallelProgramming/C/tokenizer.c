// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    27.05.2026

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>

TokenStream tokenstream_open(char* filename)
{
	//	Opens a TokenStream on the specified filename
	//	Inputs:
	//		char* filename: The filename of which we want to process into tokens with a TokenStream
	//	Outputs:
	//		A TokenStream object, which will sequentially read the file and process each buffer into a token

	FILE * fp = fopen(filename, "r");
	if(fp==NULL){ perror("Error opening file"); return NULL; }

	TokenStream ts = malloc(sizeof(*ts));

	ts -> fp = fp;
	ts -> is_loaded = 0;

	return ts;
}

void tokenstream_close(TokenStream ts)
{
	// Closes a TokenStream object. Self-explanatory and less than 3 lines, so no further explanation needed
	if(ts == NULL) return; 
	else{ fclose(ts->fp); free(ts); }
}


float process_float(TokenStream ts, char first_digit, int found_dot_initial, int* status)
{
	// Processes a float, triggered by something like +... or -... (not + ... nor - ...) or with a numeric digit.
	// Starts reading the file until it reaches a whitespace or EOF. Uses a buffer of characters on which we will
	// use the strtof function.

	// Input(s)
	// 		ts -> Tokenstream in question
	//		first_digit -> The first "digit" (not really, can be also ., - or +) which has triggered this function call
	//		found_dot_initial -> A boolean indicating whether a dot . has already been found or not
	//		status -> a pointer indicating towards an integer, used to report whether the processment (?) was failed or not
	// Output:
	//		float f -> the float converted
	//		(implicitly) status -> failure or success, if failure it changes to -1

	char* buffer = malloc(INITIAL_STRF_BUFSIZE*sizeof(char)); 

	buffer[0] = first_digit;
	int found_dot = found_dot_initial; // Reports whether the . has been already found or not.
	int multiplier_size = 1;
	int buffer_cursor = 1; // Cursor on the buffer: determines where to write on the buffer. 
	int next_char_internal_int = '0';
	unsigned char next_char_internal;
	while( ((next_char_internal_int = fgetc(ts->fp)) != EOF ) && !(isspace((unsigned char) next_char_internal_int)) )
	{ // Go on until EOF or space character
		next_char_internal = (unsigned char) next_char_internal_int;
		if(next_char_internal == '.')
		{
			if(found_dot == 1)
			{
				printf("ERROR: Unrecognizable float.\n");
				free(buffer);
				*status = -1;
				return 0.0;
			}
			found_dot = 1;
		}

		else if(!(isdigit(next_char_internal)))
		{
			printf("ERROR: Unrecognizable float (not a digit nor appropriate dot).\n");
			free(buffer);	
			*status = -1;
			return 0.0;
		}

		buffer[buffer_cursor] = next_char_internal;
		buffer_cursor++;

		// Check for buffer cursor and actual size. I do +1 to also allow the eventual '\0' in the end
		if(buffer_cursor+1 >= INITIAL_STRF_BUFSIZE*multiplier_size)
		{
			// Reallocate stuff
			multiplier_size++;
			char* tmp = realloc(buffer, multiplier_size*sizeof(char));

			if(tmp == NULL)
			{
				perror("realloc");
				free(buffer);
				*status = -1;
				return 0.0;
			}

			buffer = tmp;
		}
	}

	buffer[buffer_cursor] = '\0';

	char* endptr; 

	float f = strtof(buffer, &endptr);

	if(endptr == buffer)
	{
		printf("ERROR: String to float conversion failed. %s", buffer);
		perror("strtof");
		free(buffer);
		*status = -1;
		return 0.0;
	}

	else
	{
		free(buffer);
		return f;
	}
}

char* parse_string(TokenStream ts)
{
	// Parses a string, triggered by a " symbol. Reads the file until EOF (reports error as needs a closing ") or until " is met.
	// Basically does a while loop. Uses a buffer that can be dynamically increased, returns the length with a pointer too.
	// Input(s):  
	//		TokenStream ts -> tokenstream in question
	//		int* length -> a pointer towards an integer, used to keep track of length
	// Output(s):
	//		char* parse_string -> the parsed string, with a '\0' at the end

	char* buffer = malloc(sizeof(char)*INITIAL_STRF_BUFSIZE);
	
	int multiplier_size = 1;
	int current_size = 0;
	
	int next_char;
	int found_closing = 0;

	while( (next_char = fgetc(ts->fp) ) != EOF )
	{
		// Break if i got a "
		if( (unsigned char) next_char == '"' )
		{
			found_closing=1;
			break;
		}

		buffer[current_size] = (unsigned char) next_char;
		current_size++;

		// Increase size if necessary
		if(current_size+1 >= INITIAL_STRF_BUFSIZE*multiplier_size) // +1 to accomodate the final '\0'
		{
			multiplier_size++;
			char* tmp = realloc(buffer, sizeof(char)*INITIAL_STRF_BUFSIZE*multiplier_size);

			if(tmp==NULL){ perror("Error with realloc"); free(buffer); return NULL; }
			buffer = tmp;
		}

	}

	if(found_closing == 0)
	{
		printf("ERROR: Unclosed string. Check your script.\n");
		free(buffer);
		return NULL;
	}

	buffer[current_size] = '\0';
	return buffer;
}

int tokenstream_next(TokenStream ts, Token* token_ptr)
{
	//	Reads the next token of a TokenStream object and classify it.
	//	Inputs:
	//		TokenStream ts: TokenStream from which we want to get the next token
	//		Token* token_ptr: A pointer to a token, on which we will write the value of the read token
	//	Outputs: An integer value, indicating 0 for success, -1 for insuccess
	Token token_curr = {0};

	int current_char_int = fgetc(ts->fp);
	unsigned char current_char = (unsigned char) current_char_int;

	if (current_char_int == EOF)
	{
		// EOF
		token_curr.token = TOKEN_EOF;
	}
	// Otherwise: classify the buffer
	else
	{
		// Take a peek at the next character, 
		// for ambiguous cases such as negative numbers -1.2354 (I need to distinguish the pure minus operator from an actual number)
		// Also to make sure that the tokens are properly separated by a space (so no stuff like +[1 2 3 4])
		int next_char_int = fgetc(ts->fp);
		unsigned char next_char = (unsigned char) next_char_int;

		if (next_char_int != EOF && ungetc(next_char_int, ts->fp) == EOF) // Apparently ungetc works, while fseek does not... interesting
		{
		    perror("Error doing ungetc");
		    return -1;
		}

		// Parse weird things: numbers with a weird structure (yet allowed)
		if(
			(current_char == '+' && next_char == '.') ||  // +.1235
			(current_char == '-' && next_char == '.') ||  // -.1234
			(current_char == '+' && isdigit(next_char)) || // +1234
			(current_char == '-' && isdigit(next_char)) || // ALlows stuf like -12334
			(isdigit(current_char) && next_char == '.') || // ALlows stuff like 1.234
			(current_char == '.' && isdigit(next_char)) || // Allows stuff like .12354
			(isdigit(current_char) && isdigit(current_char)) // Allows stuff like 1234
		)
		{
			int status = 0;
			float parsed_float;
			if(current_char == '.') parsed_float = process_float(ts, current_char, 1, &status);
			else parsed_float = process_float(ts, current_char, 0, &status);

			if(status == -1)
			{
				printf("ERROR: Unparsable float.\n");
				return -1;
			}

			/* I opted to NOT use fscanf as it fails in detecting errors (e.g. detects -.434.63 as -0.434 and -0.63, while an
			error should've been appropriate. Instead, I used my own while loop logic to classify floats

			float parsed_float; 

			if (ungetc(current_char_int, ts->fp) == EOF)
			{
				perror("ungetc error:");
				return -1;
			}

			if(fscanf(ts->fp, "%f ", &parsed_float) != 1)
			{
				perror("fscanf error: ");
				return -1;
			}
			*/

			token_curr.token = TOKEN_NUMBER;
			token_curr.value.numerical_value = parsed_float;
		}

		else if(current_char == '"')
		{
			// While loop logic, terminate until EOF (in this case return error) or until another " is encountered
			// The string is a buffer that can be dynamically increased with realloc
			char * parsed_str = parse_string(ts);

			if(parsed_str == NULL)
			{
				printf("ERROR tokenizing string.\n");
				return -1;
			}

			token_curr.token = TOKEN_STRING;
			token_curr.value.string = parsed_str;
		}

		// Any other case: each character must be accompanied by a whitespace (or strings, for all that matters)
		else
		{
			if(next_char_int != EOF && !(isspace(next_char)) && !(isspace(current_char)))
			{
				printf("ERROR: Unparsable text, the following characters were found next to each other: ");
				printf("%c and %c\n", current_char, next_char);
				return -1;
			}

			if(current_char == '+')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_PLUS;
			}

			else if(current_char == '-')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_MINUS;
			}

			else if(current_char == '*')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_PROD;
			}

			else if(current_char == '<')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_CMP_LESS;
			}

			else if(current_char == '>')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_CMP_MORE;
			}

			else if(current_char == '=')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_CMP_EQ;
			}

			else if(current_char == '&')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_AND;
			}

			else if(current_char == '|')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_OR;
			}

			else if(current_char == '!')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_NOT;
			}

			else if(current_char == '$')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_DOLLAR;
			}

			else if(current_char == '@')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_MATRIX_PROD;
			}

			else if(current_char == '.')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_DOT_PROD;
			}

			else if(current_char == 'c')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_CONV;
			}

			else if(current_char == 'r')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_RESHAPE;
			}

			else if(current_char == '_')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_FLATTEN;
			}

			else if(current_char == '#')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_SHAPE;
			}

			else if(current_char == '?')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_RAND;
			}

			else if(current_char == 'R')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_RELU;
			}

			else if(current_char == 'm')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_MIN;
			}

			else if(current_char == 'M')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_MAX;
			}

			else if(current_char == 'S')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_REDUCT_SUM;
			}

			else if(current_char == 'f')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_FILL;
			}

			else if(current_char == 'p')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_PRINT;
			}

			else if(current_char == 'd')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_DUPLICATE;
			}

			else if(current_char == 's')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_SWAP;
			}

			else if(current_char == 'o')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_OVER;
			}

			else if(current_char == 'D')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_DROP;
			}

			else if(current_char == '(')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_READPGM;
			}

			else if(current_char == ')')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_WRITEPGM;
			}

			else if(current_char == '{')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_READFILE;
			}

			else if(current_char == '}')
			{
				token_curr.token = TOKEN_OPERATOR;
				token_curr.value.operator_type = OP_WRITEFILE;

			}
			else if(current_char == '[')
			{
				token_curr.token = TOKEN_LBRACKET;

			}
			else if(current_char == ']')
			{
				token_curr.token = TOKEN_RBRACKET;

			}
			else if(isdigit(current_char))
			{
				// Parse singular digits as floats by using ASCII tricks
				token_curr.token = TOKEN_NUMBER;
				token_curr.value.numerical_value = (int) (current_char - '0');
			}
			else if(isspace(current_char))
			{
				// Logic to parse the whitespaces, by calling tokenstream_next again
				if(tokenstream_next(ts, token_ptr) != 0)
				{
					// Failed for some reason
					printf("ERROR: Error in parsing token next to a whitespace\n");
					return -1;
				}
				return 0; // Otherwise it's successful and return the value as indicated

			}

			else
			{
				printf("ERROR: UNRECOGNISED TOKEN. TERMINATING.\n");
				return -1;
			}
		}

	}

	ts->t_curr = token_curr; 
	ts->is_loaded = 1;

	*token_ptr = token_curr;

	return 0;
}

int parse_tensor_recursive(TokenStream ts, tensor t, int out_shape[MAX_DIM], int* out_ndim)
{
	// 	A recursive algorithm to parse a tensor (like [ 1 2 3 ]). The idea is the following:
	//	This function is triggered by a reading of the [ token, and has two "modes":
	//	One is reading numbers only, which means that the parser has reached the most internal subtensor. Expects only other numbers
	//	One is reading subtensors, meaning that the current "level" is one of the external subtensors. Expects only other subtensors
	//	The base case is triggered by a closing bracket ] and its behaviour depends on the current mode.
	//	The recursive case is triggered by an opening bracket [, which then the current function might "expect" a shape of the
	//	child tensor and do some comparisons (if applicable)
	//	Inputs:
	//		TokenStream ts: TokenStream from which we want to parse the tensor
	//		tensor t: Pointer to a tensor, on which we will write the data inside
	//		out_shape[MAX_DIM]: Shape of the final output (of the CURRENT level in the recursive nesting)
	//		out_ndim: Same as above but number of dimensions
	//	Ouputs: An integer indicating either success or failure. -1 for failure, 0 for success

    Token token_tensor_internal;

    bool saw_number = 0;
    bool saw_subtensor = 0; // To keep track whether I saw numbers or subtensors, to reject mixing (like [1 2 [ 2 3 4 ] ])

    int expected_child_shape[MAX_DIM];
    int expected_child_ndim; // For recursive cases (called multiple times

    bool expected_child_is_defined = 0;

    int element_count = 0; // to count how many elements (or subtensors) added in this recursive cycle

    while(tokenstream_next(ts, &token_tensor_internal) == 0)
    {
   	// Total mistake: EOF
    	if(token_tensor_internal.token == TOKEN_EOF)
    	{
    		printf("ERROR: EOF during tensor parsing. Exploding\n");
    		return -1;
    	}
    // Recursive case: encounter another [
        if(token_tensor_internal.token == TOKEN_LBRACKET)
        {
            saw_subtensor = 1;

            if(saw_number == 1)
            {
                printf("ERROR: Illegal tensor (mixed).\n");
                return -1;
            }

            int child_shape[MAX_DIM];
            int child_ndim; 

            if(parse_tensor_recursive(ts, t, child_shape, &child_ndim) != 0)
            {
                printf("Failed recursive parsing of the tensor.\n");
                return -1;
            }

            if(!expected_child_is_defined)
            {
                // If expected child is not defined yet, then define it NOW with the first child being explored
                expected_child_ndim = child_ndim;
                for(int i=0; i<expected_child_ndim; i++)
                {
                    expected_child_shape[i] = child_shape[i];
                }

                expected_child_is_defined=1;
            }

            else
            {
                // Compare your expectations with your children
                bool cond = 0;
                if (expected_child_ndim != child_ndim) cond=cond || 1;

                for(int i = 0; i<expected_child_ndim; i++)
                {
                    if(expected_child_shape[i] != child_shape[i]) cond = cond || 1;
                }

                if(cond){ 
                    printf("ERROR: Illegal tensor (ragged, i.e. unmatching shapes\n");
                    return -1;
                }
            }

            element_count++;

        }

        else if(token_tensor_internal.token == TOKEN_NUMBER)
        {
            saw_number = 1;

            if(saw_subtensor == 1)
            {
                printf("ERROR: Illegal tensor (mixed).\n");
                return -1;
            }

            // logic to insert numbers into array and expand dimension if needed
            if(t->current_capacity >= t->max_capacity)
            {
                t->max_capacity*=2;
                float* tmp = realloc(t->data, sizeof(float)*t->max_capacity);

                if(tmp == NULL){ printf("ERROR in reallocating the data array\n"); return -1;}
                else t->data=tmp;
            }

            t->data[t->current_capacity] = token_tensor_internal.value.numerical_value;
            t->current_capacity++;

            element_count++;

        }
    // Base case: encounter ]

        else if(token_tensor_internal.token == TOKEN_RBRACKET)
        {
            if (saw_number == 0 && saw_subtensor == 0){ printf("ERROR: Empty tensor (or subtensors) not allowed\n"); return -1; }
            
            else if(saw_number == 0 && saw_subtensor == 1)
            {
                // In this case i'm closing a subtensor

                if(expected_child_ndim + 1 > MAX_DIM)
                {
                    printf("Tensor exceeds maximum defined dimensions.\n");
                    return -1;
                }

                out_shape[0] = element_count;
                for(int i = 0; i < expected_child_ndim; i++)
                {
                    out_shape[i+1] = expected_child_shape[i]; // Copy the ndims of the children
                }

                *out_ndim = expected_child_ndim + 1;
                return 0; 
            }

            else if(saw_number == 1 && saw_subtensor == 0)
            {
                // In this case i'm closing the most internal subtensor
                *out_ndim = 1;
                out_shape[0] = element_count;
                return 0;
            }

            else
            {
                // Impossible
                printf("Error: anomaly occurred.\n");
                return -1;
            }
        }

    }
}

tensor parse_tensor(TokenStream ts) // depth is the current "dimension depth" of the tensor. basically will determine ndim
{
	//	Function to parse a tensor from the TokenStream ts.
	//	Input:
	//		TOkenstream ts: Tokenstream
	//	Ouput: the resulting tensor. If failure, a NULL pointer is returned

    // Next hard part: once this is done, the project should be FINISHED. Gets called every time [ is read, recursive algorithm!

    // Idea: if number, it's a 1D vector and search only for numbers from now on. Otherwise if i meet anything else (except for ]), it's a big mistake
    // Otherwise if it's [, then go back to the recursive case. Once the recursive case is over, ONLY search for ]
    tensor t = malloc(sizeof(*t));
    t->on_mmap = 0;
    t->data = malloc(sizeof(float)*INITIAL_FLOAT_BUFSIZE);
    t->max_capacity = INITIAL_FLOAT_BUFSIZE;
    t->current_capacity = 0;
    t->ndim = 0;
    t->total_references=0;

    int final_shape[MAX_DIM];
    int final_ndim;

    if(parse_tensor_recursive(ts, t, final_shape, &final_ndim) != 0)
    {
        printf("Error parsing tensor. Sorry!\n");
        free(t->data);
        free(t);
        return NULL;
    }

    t->ndim = final_ndim;

    for(int i=0; i < t->ndim; i++)
    {
        t->shape[i] = final_shape[i];
    }

    return t;
}

