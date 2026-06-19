#include "tokenizer.h"
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    TokenStream ts = tokenstream_open("./examples/random_matmul.tensorforth");

    if (ts == NULL) {
        return 1;
    }

    Token token_cursor;

    while (tokenstream_next(ts, &token_cursor) == 0)
    {
        if (token_cursor.token == TOKEN_EOF)
        {
            printf("End of file. Ending.\n");
            break;
        }

        printf("Token of type %d\n", (int)token_cursor.token);

        if (token_cursor.token == TOKEN_NUMBER)
        {
            printf("\t%f\n", token_cursor.value.numerical_value);
        }
    }

    tokenstream_close(ts);
}