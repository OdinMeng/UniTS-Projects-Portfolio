#include "tensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h>

int main(void)
{

	float data_alt[] = {
		1, 2,
		3, 4
	};

	tensor shape_test = malloc(sizeof(*shape_test));
	shape_test -> ndim = 2;
	shape_test -> shape[0] = 2;
	shape_test -> shape[1] = 2;
	shape_test -> data = data_alt;

	print_tensor(shape_test);
	print_tensor(shape_test);
	print_tensor(shape_test);


	float data[] = {
		3, 3
	};

	tensor shape_t = malloc(sizeof(*shape_t));

	shape_t -> shape[0] = 2;
	shape_t -> ndim = 1;
	shape_t -> data = data; 

	print_tensor(shape_t);
	print_tensor(shape_t);

	tensor rand_t; 
	rand_t = rand_tensor(shape_t);

	print_tensor(rand_t);
	print_tensor(rand_t);

	free(rand_t->data);
	free(rand_t);
	free(shape_test);
	free(shape_t);
	return 0;
}