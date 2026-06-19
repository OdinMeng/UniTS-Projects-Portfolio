#include "tensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h>

int main(void)
{

	float data_1[] = {
		-1, 10, 1.23,
		1.45, 0.23, 1.23,
		1.10, 1.23, 1.45
	};

	float data_2[] = {
		9, 9, 9,
		9, 9, 9,
		9, 9, 9,
	};


	float data_m[] = {
		0, 0, 1,
		0, 1, 0,
		1, 0, 0
	};


	tensor t1 = malloc(sizeof(*t1));
	t1 -> ndim = 2;
	t1 -> shape[0] = 3;
	t1 -> shape[1] = 3;
	t1 -> data = data_1;

	tensor t2 = malloc(sizeof(*t2));
	t2 -> ndim = 2;
	t2 -> shape[0] = 3;
	t2 -> shape[1] = 3;
	t2 -> data = data_2;

	tensor tm = malloc(sizeof(*tm));
	tm -> ndim = 2;
	tm -> shape[0] = 3;
	tm -> shape[1] = 3;
	tm -> data = data_m;


	print_tensor(t1);
	print_tensor(t2);
	print_tensor(tm);

	tensor t_res; t_res = select_a_or_b(tm, t1, t2);

	print_tensor(t_res);

	free(t_res -> data);
	free(t_res);

	free(t1);
	free(t2);
	free(tm);


	/*
	float data_shape[] = {
		2, 3
	};

	tensor shape_arr = malloc(sizeof(*shape_arr));
	shape_arr -> ndim = 1;
	shape_arr -> shape[0] = 2;
	shape_arr -> data = data_shape; 

	print_tensor(shape_arr);

	float data_v[] = {
		1, 2
	};

	tensor v = malloc(sizeof(*v));
	v -> ndim = 1;
	v -> shape[0] = 2;
	v -> data = data_v;

	print_tensor(v);

	tensor t_fill; t_fill = fill_shape_value(shape_arr, v);

	print_tensor(t_fill);

	free(t_fill->data);
	free(t_fill);

	free(shape_arr);
	free(v);
	*/

	return 0;
}