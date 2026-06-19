// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    09.06.2026

// This is the part of the code where I define all of the operation on tensors

#include "tensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <sys/mman.h>

// Define parallelization thresholds; by doing manual benchmarks I observed that parallelization
// starts improving performances only for some functions, and by certain "size" thresholds. 
// To obtain these values I did various manual benchmarks by manually limiting the cores numbers
// The benchmarks are (more or less) detailed in the comments for the individual functions


#ifndef PARALLELIZATION_THRESHOLDS

#define PARALLELIZATION_THRESHOLDS
#define PARALLELIZATION_THRESHOLD_MATMUL 40000000
#define PARALLELIZATION_THRESHOLD_DOT 150000
#define PARALLELIZATION_THRESHOLD_CONV 15000000
#define PARALLELIZATION_THRESHOLD_FILL 5300000

#endif

// Part 7: useful stuff (done first because this is SUPER important for debugging)

void explore_tensor_recursive(tensor t, int32_t current_dim, int32_t* pos)
{
	// Recursive algorithm to print a tensor according to its shape; the idea is to iterate over the shape of the tensor. NOT parallelized as this depends on the pointer pos
	// Input(s):
	//		tensor t: Tensor of which we want to read the tensor
	//		int32_t current_dim: Current dimension, for the recursive part of the code
	//		int32_t* pos: A pointer which keeps track fo the current position in the array t
	// Output(s):
	//		None, everything is printed

	printf("[ ");

	// Iterate on the shape of the current dimension
	for(int32_t i=0; i < t->shape[current_dim]; i++)
	{	
		// Base case: we are at the last dimension and we will start printing everything
		if(current_dim == t->ndim-1)
		{
			printf("%f", t->data[*pos]);
			(*pos)++;
		}
		
		// Recursive case: Pass to the next dimension	
		else
		{
			explore_tensor_recursive(t, current_dim+1, pos);
		}

		// print whitespaces for all elements except the last one
		if(i < t->shape[current_dim]-1)
		{
			printf(" ");
		}
	}

	printf(" ]");
}


void print_tensor(tensor t)
	// Function to print the shape and data of a tensor.
	// Input(s):
	//		tensor t: Tensor of which we want to print info
	// Output(s):
	//		None, everything is printed

{

	printf("Tensor(ndim=%d, shape=[ ", t->ndim);
	
	for(int32_t i=0; i<t->ndim; i++)
	 	{
		 	printf("%d", t->shape[i]);
		 	printf(" ");
	 	}

 	printf("], data=");

 	int32_t pos = 0;

	explore_tensor_recursive(t, 0, &pos);

 	printf(")\n");

}

bool check_is_shape_array(tensor shape_array)
{
	// Checks whether a tensor represents a shape array, that is all the following requisites are respected:
	// (1) Tensor has only 1 dimension
	// (2) Along the first dimension, there can be only at most MAX_DIM elements
	// (3) Each element is a positive integer
	// Input(s):
	//		tensor shape_array: Tensor which represents a shape tensor.
	// Output(s):
	//		bool is_shape_array: Result of the checks

	bool is_shape_array = 1; 

	if((shape_array->shape)[0] > MAX_DIM){ is_shape_array = 0;} // Checks (2)

	if(shape_array->ndim > 1){ is_shape_array = 0; } // Checks (2)

	if(is_shape_array) // The following lines might be potentially be already skippable
		{
		//#pragma omp parallel for reduction(&&: is_shape_array)
			// Removed parallelization for the following reasons:
			// 1. Not necessary as shape[0] is a very small number, at most MAX_DIM
			// 2. Causes slowdown as the cost for running multithreading is very expensive relative to the gained advantage
		for(int32_t i=0; i<(shape_array->shape[0]); i++)
		{
			is_shape_array = is_shape_array && ((int)(shape_array->data)[i] == (shape_array->data)[i]) && (shape_array->data)[i]>0;
				// Checks (3)
		}
	}

	return is_shape_array;
}

tensor rand_tensor(tensor shape_array)
{
	// Randomly generates a tensor with rand() function. NOT parallelized because the function is not thread safe
	// Input(s):
	//		tensor shape_array: Tensor which represents the shape of the tensor we want to obtain a random tensor.
	//			N.B. Many important assumptions! (1) Must be 1D, (2) Can have AT MOST MAX_DIM elements inside, (3) Must contain only non-negative integer values inside
	// Output(s):
	//		tensor t: Tensor with shape shape_array which contains randomly generated elements


	// 0. check shape array
	if(!check_is_shape_array(shape_array))
	{
		printf("ERROR: tensor shape_array is not a shape array. Returning with NULL pointer.\n");
		return NULL;
	}

	// 1: initialize the tensor according to the shape
	tensor t = malloc(sizeof(*t));
	t->ndim = *(shape_array->shape); // access the 0-th element

	// iterate over the shape tensor to determine tensor's shape
	for(int32_t i=0; i<t->ndim; i++) 
		t->shape[i] = (int32_t)(shape_array->data)[i];

	// fill auxiliary data
	t->on_mmap=0;

	// 2: fill the tensor
	int32_t total = 1;
	for(int32_t n=0; n<t->ndim; n++) total *= t->shape[n];

	t->data = (float*) malloc(sizeof(float)*total);
	if(!t->data)
	{
		free(t); printf("ERROR: Failed malloc. Returning with NULL pointer");
		return NULL;
	}

	for(int32_t i=0; i<total; i++) {float x=(float)rand() / RAND_MAX; t->data[i] = x; }

	// 3: return the pointer
	return t;
}

// Part 1: Pointwise arithmetic operations (+, -, *, ...)
bool check_same_shapes(tensor t1, tensor t2)
{
	// Function to check whether two tensors have the shapes or not. Useful for some functions where this assumption is necessary (e.g. pointwise operations between tensors)
	// Input(s):
	//		tensor t1: Tensor 1
	//		tensor t2: Tensor 2
	// Output(s):
	//		val: A boolean value indicating whether the requirement has been met or not.

	if(t1->ndim != t2 -> ndim) return 0;

	for(int i=0; i<t1->ndim; i++) // Can this be parallelized? maybe. Should I? IDK man, NDIM is equal to 2 anyways
	{
		if(t1->shape[i] != t2->shape[i]) return 0;
	}

	return 1;

	// Potential parallelization:
	// 	bool cond=1;
	//	#pragma omp parallel for reduction(&&:same)
	//	for(int i=0; i<t1->ndim; i++) same = same && (t1->shape[i] == t2->shape[i]);
	//	return same;
	// NOT done as MAX_DIM is low anyways (2)
}

int32_t get_total_shape(tensor t)
{
	// Gets total number of elements according to a tensor's shape. Basically does the product of its shape
	// Input(s):
	//		tensor t: tensor 
	// Output(s):
	//		int_32t total: Product of the shape array

	int32_t total = 1;
	for(int i=0; i<t->ndim; i++) total *= t->shape[i];
	return total;
}

// Define sum, subtraction and multiplication functions as pointwise operators
static float add_f(float a, float b) { return a + b; }
static float sub_f(float a, float b) { return a - b; }
static float mul_f(float a, float b) { return a * b; }

tensor tensors_op_binary(tensor t1, tensor t2, binary_op op)
{
	// Performs a binary operator on two tensors with same shape
	// Input(s):
	//		tensor t1: tensor 1 
	//		tensor t2: tensor 2 
	//		IMPORTANT ASSUMPTION: t1 and t2 must have the same shape, checked with the auxiliary function check_same_shapes.
	// Output(s):
	//		tensor t_res: Resulting tensor of op(t1, t2)

	if(!check_same_shapes(t1, t2)){ printf("ERROR: Tensors t1 and t2 have different shapes/ndims.\n"); return NULL; }
	tensor t_res = malloc(sizeof(*t_res));
	int32_t total = get_total_shape(t1); 

	// fill necessary info
	t_res -> on_mmap = 0; 
	memcpy(t_res->shape, t1->shape, sizeof(t_res->shape));
	t_res -> ndim = t1 -> ndim; 
	t_res -> data = malloc(sizeof(float)*total);

	//#pragma omp parallel for
		// Decided to remove parallelization since the performances were always degraded with parallelization, 
		//even with large tensors
	for(int i=0; i<total; i++)
	{
		(t_res->data)[i] = op((t1->data)[i], (t2->data)[i]);
	}

	return t_res;
}

tensor sum_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, add_f);
}

tensor diff_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, sub_f);
}

tensor mul_tensors_pointwise(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, mul_f);

}

// Part 2: this is literally the same as part 1 but I cast stuff like a<b et cetera
static float less_f(float a, float b) { return (float)a<b; }
static float more_f(float a, float b) { return (float)a>b; }
static float eq_f(float a, float b) { return (float)a==b; }

tensor cmp_less_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, less_f);
}

tensor cmp_more_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, more_f);
}

tensor cmp_eq_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, eq_f);
}

// Part 3: literally the same as before, just add more checks (i.e. must check for binary values in tensors). Only exception is !, which is an unary operation and is handled differently

static float and_f(float a, float b) { return (float) ((bool) a & (bool) b); }
static float or_f(float a, float b) { return (float) ((bool) a | (bool) b); }

bool check_binary_values(tensor t)
{
	// Checks whether a tensors contains only binary values or not (i.e. 1, 0)
	// Input(s):
	//		tensor t: tensor to check
	// Output(s):
	//		bool is_binary: Result, 1 or 0 (True/False)
	bool is_binary = 1;
	int32_t total = get_total_shape(t);

	// #pragma omp parallel for reduction(&&: is_binary) // parallelize with a reduction
		// removed since worsened performances here as well, for some reason
	for(int i=0; i<total; i++)
	{
		is_binary = is_binary & ( (t->data)[i] == (float)1 | (t->data)[i] == (float)0);
	}

	return is_binary;
}

tensor tensors_op_binary_logical(tensor t1, tensor t2, binary_op op)
{
	if(!(check_binary_values(t1) && check_binary_values(t2)))
	{
		printf("ERROR: Tensor t1 or t2 do not have binary values. Returning NULL pointer.\n");
		return NULL;
	}
	return tensors_op_binary(t1, t2, op);
}

tensor or_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary_logical(t1, t2, or_f);
}

tensor and_tensors(tensor t1, tensor t2)
{
	return tensors_op_binary_logical(t1, t2, and_f);
}

tensor tensor_unary_op(tensor t, unary_op op)
{
	// The evil brother of tensors_binary_op, basically does the same thing but on single tensors with unary operators
	// Input(s):
	//		tensor t: tensor
	// Output(s):
	//		tensor t_res: Resulting tensor of op(t)
	// Very useful for some unary oprations on tensors (like NOT, ReLU, and so on...)

	tensor t_res = malloc(sizeof(*t_res));
	int32_t total = get_total_shape(t); 

	// fill necessary info
	t_res -> on_mmap = 0; 
	memcpy(t_res->shape, t->shape, sizeof(t->shape));
	t_res -> ndim = t -> ndim; 
	t_res -> data = malloc(sizeof(float)*total);

	// start calculating!!! THIS CAN BE PARALLELIZED EASILY, WILL DO IT LATER
	//#pragma omp parallel for
		// Decided to remove parallelization here as well for the same reason with binary operations
	for(int i=0; i<total; i++)
	{
		(t_res->data)[i] = op((t->data)[i]);
	}

	return t_res;
}

static float not_f(float x) { return (float)!x; }


tensor not_tensor(tensor t)
{
	if(!(check_binary_values(t)))
	{
		//There should be an error, but will make it a warning instead since it can handle errors for some reason
		printf("WARNING: Tensor t does not have binary values (in some cases). All non-zero values will be treated as 1 (TRUE), and 0 will be treated as 0 (FALSE).\n");
	}

	return tensor_unary_op(t, not_f);
}

// Part 4: selection
// Two ways: either hard-code it or make a function that returns the parametrized function f_{a,b}(x).
// I'm gonna hard code it... the other way seems too complicated (possible but probably NOT worth it)

tensor select_a_or_b(tensor t, tensor a, tensor b)
{
	// Implements the ? operator in a hard-coded fashion (due to technical limitations of C)
	// Input(s):
	//		tensor t: tensor
	//		tensor a: tensor which "substitutes" 1
	// 		tensor b: tensor which "replaces" 0
	// Output(s):
	//		tensor t_res: Resulting tensor of op(t)
	// Very useful for some unary oprations on tensors (like NOT, ReLU, and so on...)

	if(!(check_binary_values(t)))
	{
		// Warning
		printf("WARNING: Tensor m (of m?a:b) does not have binary values (in some cases). All non-ones will be treated as zeros.\n");
	}

	// Check that all of the shapes are equal to each other
	if(!((check_same_shapes(a, b)) && (check_same_shapes(a, t)))) // Two are enough by transitive property; a = b & a = t => t = b
	{
		printf("ERROR: Shape mismatch in operator $\n");
		return NULL;
	}

	tensor t_res = malloc(sizeof(*t_res));
	int32_t total = get_total_shape(t); 

	// fill necessary info
	t_res -> on_mmap = 0; 
	memcpy(t_res->shape, t->shape, sizeof(t->shape));
	t_res -> ndim = t -> ndim; 
	t_res -> data = malloc(sizeof(float)*total);

	//#pragma omp parallel for
		// Removed for same reason as before
	for(int i=0; i<total; i++)
	{
		bool q = ((t->data)[i] == (float)1);
		(t_res->data)[i] = q * (a->data[i]) + (1-q) * (b->data[i]); 
		// Branchless programming because in most cases it should be something like a 50/50 coin toss, OFC depends on what the users will pass
	}

	return t_res;
}

// Part 8: Other pointwise operators
static float relu_f(float x) { bool q = (x > 0); return q*x; } // Branchless programming, since usually x can be any value...
static float max_f(float x, float y) { bool q = (x>y); return q*x + (1-q)*y; }
static float min_f(float x, float y) { bool q = (x<y); return q*x + (1-q)*y; }

tensor relu_tensor(tensor t)
{
	return tensor_unary_op(t, relu_f);
}

tensor min_tensor(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, min_f);
}

tensor max_tensor(tensor t1, tensor t2)
{
	return tensors_op_binary(t1, t2, max_f);
}

// Part 9: Reduction sum
tensor reduct_sum_tensor(tensor t)
{
	// Implements the S(...) operator
	// Input(s):
	//		tensor t: tensor
	// Output(s):
	//		tensor t_res: Resulting sum of the tensor as a 1D tensor

	tensor t_res = malloc(sizeof(*t_res));
	int32_t total = get_total_shape(t); 

	// fill necessary info
	t_res -> on_mmap = 0; 
	(t_res -> shape)[0] = 1;
	t_res -> ndim = 1; 
	t_res -> data = malloc(sizeof(float)*1); // Only 1 element

	double sum = 0;

	// calculate sum and save on t_res
	// #pragma omp parallel for reduction(+: sum)
		// Removed parallelization for reasons same as above... performances keep degrading even on very big matrixes
	for(int i=0; i<total; i++)
	{
		sum += (double)(t->data)[i];
	} // Use doubles for better precision (with float some slight numerical errors may be commited)

	*(t_res -> data)=(float)sum;

	return t_res;
}

// Part 6: Shape operators
tensor reshape_tensor(tensor shape_array, tensor t)
{
	// Implements the reshape operator
	// Input(s):
	//		tensor t: tensor
	//		tensor shape_array: shape array-tensor
	// Output(s):
	//		tensor t_new: Resulting tensor

	// Pre-preliminary step: check that the shape array is a shape array with non negative integer values
	if(!check_is_shape_array(shape_array))
	{
		printf("ERROR: Shape array provided is not a shape array.");
		return NULL;
	}


	// Preliminary step: check for compatibility
	int32_t total_t = get_total_shape(t);
	int32_t total_shape = 1;

	// this can be parallelized but not done since shape_array's shape is expected to be small enough
	for(int i=0; i < (shape_array->shape)[0]; i++) total_shape *= (shape_array->data)[i];

	if(total_t != total_shape)
	{
		printf("ERROR: New shape and current array's shape are not compatible. Creating a black hole in your computer.\n");
		return NULL;
	}

	// If everything is OK, actually do the reshape (just pass the data...)

	tensor t_new = malloc(sizeof(*t_new));
	int32_t total = get_total_shape(t); 

	// fill necessary info
	t_new -> on_mmap = 0; 

	// Not parallelized since it's expected to be a small value
    for (int32_t i = 0; i < shape_array->shape[0]; i++) 
    {
        t_new->shape[i] = (int32_t)shape_array->data[i];
    }

	t_new -> ndim = shape_array -> shape[0]; 
	t_new -> data = malloc(sizeof(float)*total);

	memcpy(t_new->data, t->data, sizeof(float)*total);

	return t_new;
}

tensor flatten_tensor(tensor t)
{
	// Implements the flatten operator. Basically just calls the reshape operatro with an ad-hoc shape array.
	// Input(s):
	//		tensor t: tensor
	// Output(s):
	//		tensor t_new: Resulting tensor

	tensor shape_array = malloc(sizeof(*shape_array));
	shape_array->ndim = 1;
	shape_array->shape[0]=1;
	shape_array->data = malloc(sizeof(float)*1);

	*(shape_array->data) = get_total_shape(t);

	tensor t_new;
	t_new = reshape_tensor(shape_array, t);

	free(shape_array->data);
	free(shape_array);

	return t_new;

}

tensor get_tensor_shape(tensor t)
{
	// Returns the shape of a tensor as a 1D tensor
	// Input(s):
	//		tensor t: tensor
	// Output(s):
	//		tensor shape_array: shape of t
	tensor shape_array = malloc(sizeof(*shape_array));
	shape_array->ndim = 1;
	shape_array->shape[0]=t->ndim;
	
	shape_array->data = malloc(sizeof(float)*(shape_array->shape[0]));

	// can be easily parallelizable, but won't since MAX_DIM=2 anyways
    for (int32_t i = 0; i < t->ndim; i++) {
        shape_array->data[i] = (float)t->shape[i];
    }

	return shape_array;
}

// Part 5: specific operations

tensor product_tensors(tensor t1, tensor t2)
{
	// Returns the matrix multiplication between t1 and t2. Implements a parallelized version of the transpose matrix multiplication
	// Input(s):
	//		tensor t1, tensor t2: tensors of which we want to do the matrix multiplication
	//		N.B. Assumes that both of them have ndim=2 and the shapes are compatible (i.e. of type (nxp)(pxm))
	// Output(s):
	//		tensor t_res: Tensor result, with shape n x m (from n x p, p x m)

	// Step 1: check for matrixes
	if(!(t1->ndim == 2 && t2->ndim == 2))
	{
		printf("ERROR: One between t1, t2 is not a 2D matrix, failing matrix product\n");
		return NULL;
	}
	if( ( t1->shape[1] != t2->shape[0] ))
	{
		printf("ERROR: Incompatible matrix shapes. Returning NULL pointer...\n");
		return NULL;
	}

	int32_t m = t1->shape[0];
	int32_t p = t1->shape[1];
	int32_t n = t2->shape[1];

	// Step 2: create resulting array
	tensor t_res = malloc(sizeof(*t_res));

	// fill necessary info
	t_res -> on_mmap = 0; 
	(t_res -> shape)[0] = m;
	(t_res -> shape)[1] = n;
	t_res -> ndim = 2; 
	t_res -> data = malloc(sizeof(float)*(m*n)); // We have m x n elements

	// transpose the other matrix for cache friendliness
	float * t2_T = malloc(sizeof(float)*p*n);

	// Removed parallelization since it slighly degrades performance
	for(int i=0; i<p; i++)
	{
		for(int j=0; j<n; j++) t2_T[j*p+i] = t2->data[i*n+j];
	}

	// fill zeros of the resulting matrix
	#pragma omp parallel for if(n*m*p > PARALLELIZATION_THRESHOLD_MATMUL)
		//Parallelized only if it has to work on data with more than 40 million elements
		// Estimated this value by running a lot of tests, in particular with two 300x300, 350x350 and 400x400 matrixes (with 350x350 the performances were the same)
		// and using hyperfine to time performances. Unfortunately perf could not have been used due to VM-related issues
	for (int32_t i = 0; i < m; i++) 
	{
		for (int32_t j = 0; j < n; j++) 
		{
			float sum = 0.0f;

			for (int32_t k = 0; k < p; k++)
				sum += t1->data[i*p + k] * t2_T[j*p + k];

			t_res->data[i*n + j] = sum;
	    }
	}
	free(t2_T);


	return t_res;
}

tensor dot_product_tensors(tensor t1, tensor t2)
{
	// Returns the dot product between 1D tensors (vectors)
	// Input(s):
	//		tensor t1, tensor t2: tensors of which we want to do the inner product
	//		N.B. Assumes that both of them have ndim=1
	// Output(s):
	//		tensor t_res: Result of the dot product as a 1D tensor with only one element

	// Step 1: check if both tensors are 1D tensors or not
	if(!(t1->ndim == 1 && t2->ndim == 1))
	{
		printf("ERROR: One between t1, t2 is not a 1D tensor, failing dot product between tensors\n");
		return NULL;
	}

	// Also check if their shapes are the same or not
	if( (t1->shape)[0] != (t2->shape)[0])
	{
		printf("ERROR: Incompatible vectors. No.\n");
		return NULL;
	}

	// Step 2: create resulting array
	tensor t_res = malloc(sizeof(*t_res));

	// fill necessary info
	t_res -> on_mmap = 0; 
	(t_res -> shape)[0] = 1;
	t_res -> ndim = 1; 
	t_res -> data = malloc(sizeof(float)*1); // Only 1 element

	// Calculate the dot product
	int32_t total = get_total_shape(t1); 
	double dot_product = 0;

	if(total > PARALLELIZATION_THRESHOLD_DOT)
	{
		#pragma parallel for reduction(+:dot_product)
			// Parallelized only if the two vectors have at least 150k elements. 
			// Threshold value estimated by multiple experimentations, with vectors of various sizes (122.5k, 140k and 160k)
			// Observed that the performances started matching at around 140k
		for(int32_t i = 0; i<total; i++) dot_product += (double) ((t1->data)[i] * (t2->data)[i]);
	}
	else
	{
		for(int32_t i = 0; i<total; i++) dot_product += (double) ((t1->data)[i] * (t2->data)[i]);
	}

	(t_res->data)[0] = (float)dot_product;

	return t_res;
}

tensor conv_tensors(tensor k, tensor t)
{
	// Returns the convolution operator with SAME output shape
	// between a matrix and an arbitrary kernel matrix using
	// zero-padding
	//
	// Input(s):
	//		tensor t: input matrix
	//		tensor k: kernel matrix
	//
	//		N.B. Assumes that both tensors have ndim=2
	//
	// Output(s):
	//		tensor t_res: Convolution result

	// Step 0: check requirements
	if(!(t->ndim == 2 && k->ndim == 2))
	{
		printf("ERROR: One between t, k is not a 2D matrix, failing convolution operator\n");
		return NULL;
	}

	if((k->shape)[0] <= 0 || (k->shape)[1] <= 0)
	{
		printf("ERROR: Invalid kernel shape.\n");
		return NULL;
	}

	// Step 1: Create the result array
	tensor t_res = malloc(sizeof(*t_res));
	int32_t total = get_total_shape(t);

	// fill necessary info
	t_res->on_mmap = 0;
	memcpy(t_res->shape, t->shape, sizeof(t_res->shape));
	t_res->ndim = t->ndim;
	t_res->data = malloc(sizeof(float) * total);

	// Step 2: Calculate padding size
	// SAME convolution => preserve original dimensions
	int32_t kh = (k->shape)[0];
	int32_t kw = (k->shape)[1];

	int32_t pad_h = kh / 2;
	int32_t pad_w = kw / 2;

	// Step 3: Calculate the convolution

	int32_t n = t_res->shape[0];
	int32_t m = t_res->shape[1];

	// Parallelizes if only we require enough computations, in particular:
	// By doing some benchmarks we got that the parallelized version gets better with 1500x1500, 3x3 and
	// 1000x1000, 5x5 matrixes and the performances are relatively the same for 1000x1000, 3x3.
	// Therefore the estimated cutoff value is 15 000 000

	if(n*m*k->shape[0]*k->shape[1] >= PARALLELIZATION_THRESHOLD_CONV)
	{
		#pragma omp parallel for
		for(int32_t I = 0; I < n * m; I++)
		{
			int32_t i = (int32_t) I / m;
			int32_t j = I % m;

			// (i,j) is the "center" of the current convolution block

			float conv_result = 0;

			// potentially parallelizable, but prefer to parallelize the outer cycle as there are "more jobs" to do there
			for(int32_t a = 0; a < kh; a++)
			{
				for(int32_t b = 0; b < kw; b++)
				{
					// (a,b) is the kernel coordinate
					// Convert to offset from center

					int32_t dx = a - pad_h;
					int32_t dy = b - pad_w;

					int32_t x = i + dx;
					int32_t y = j + dy;

					if(x < 0 || y < 0 || x >= n || y >= m)
					{
						continue; // equivalent to adding 0
					}

					else
					{
						conv_result +=
							((t->data)[x * m + y] *
							 (k->data)[a * kw + b]);
					}
				}
			}
			(t_res->data)[I] = conv_result;
		}
	}

	else
	{
		for(int32_t I = 0; I < n * m; I++)
		{
			int32_t i = (int32_t) I / m;
			int32_t j = I % m;

			// (i,j) is the "center" of the current convolution block

			float conv_result = 0;

			// potentially parallelizable, but prefer to parallelize the outer cycle as there are "more jobs" to do there
			for(int32_t a = 0; a < kh; a++)
			{
				for(int32_t b = 0; b < kw; b++)
				{
					// (a,b) is the kernel coordinate
					// Convert to offset from center

					int32_t dx = a - pad_h;
					int32_t dy = b - pad_w;

					int32_t x = i + dx;
					int32_t y = j + dy;

					if(x < 0 || y < 0 || x >= n || y >= m)
					{
						continue; // equivalent to adding 0
					}

					else
					{
						conv_result +=
							((t->data)[x * m + y] *
							 (k->data)[a * kw + b]);
					}
				}
			}
			(t_res->data)[I] = conv_result;
		}
	}

	return t_res;
}

// Part 10: fillings
tensor fill_shape_value(tensor v, tensor shape_array)
{
	// Creates a tensor with shape indicated as in shape_array, with values indicated in the tensor v
	// Input(s):
	//		tensor shape_array: shape array
	//		tensor v: tensor containing the values we want to copy inside the new tensor
	// Output(s):
	//		tensor t: New tensor!

	if(!check_is_shape_array(shape_array))
	{
		printf("ERROR: Shape array given is not a shape array. Returning NULL pointer\n");
		return NULL;
	}

	// 1: initialize the tensor according to the shape
	tensor t = malloc(sizeof(*t));
	t->ndim = *(shape_array->shape); // access the 0-th element

	// iterate over the shape tensor to determine tensor's shape

	// not parallelized since ndim <= MAX_DIM = 2 anyways lol
	for(int32_t i=0; i<t->ndim; i++) 
		t->shape[i] = (int32_t)(shape_array->data)[i];

	// fill auxiliary data
	t->on_mmap=0; 

	// Fill the tensor with data, with a paralellized for
	int32_t total = get_total_shape(t);
	t->data = malloc(sizeof(float)*total);

	int32_t total_v = get_total_shape(v);

	// The operation is parallelized only for shapes with at least 5 300 000 elements.
	// Value estimated by doing benchmarks manually (setting OMP_NUM_THREADS and measuring the execution times)

	if(total_v >= PARALLELIZATION_THRESHOLD_FILL)
	{
		#pragma omp parallel for 
		for(int32_t i = 0; i < total; i++)
		{
			(t->data)[i] = (v->data)[i % total_v];
		}
	}
	else
	{
		for(int32_t i = 0; i < total; i++)
		{
			(t->data)[i] = (v->data)[i % total_v];
		}	
	}
	
	return t;
}

// utility function
int free_tensor(tensor t)
{
	// 	Frees a tensor, according to whether it's allocated with malloc or mmap. This assumes that the tensors' total references had been checked beforehand
	// 	Input:
	//		tensor t, the tensor
	//	Output: An integer value, basically always 0. Idk why to be honest, maybe I wanted to add some error values but I ended up...
	// 	not doing that

	// Preliminary check: total references
	if(t->total_references > 0)
	{
		printf("WARNING WARNING: DEALLOCATING A TENSOR THAT STILL HAS REFERENCES IN STACK. EXPECT MEMORY LEAKS OR COMPUTER IMPLOSION\n");
	}


	if(t->on_mmap)
	{
		// LOGIC FOR MMAP DEALLOCATION HERE
		munmap(t->data, t->mmap_size);
		free(t);
		return 0;
	}	

	else
	{
		// LOGIC FOR MALLOC DEALLOCATION HERE
		free(t->data);
		free(t);
		return 0;
	}
}