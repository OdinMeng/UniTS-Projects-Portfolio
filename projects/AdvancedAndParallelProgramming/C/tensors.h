// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    15.05.2026

// This is the part of the code where I define the structure of the tensors

// CONSIDERED FINISHED AS OF 05.05.2026

#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

#ifndef MAX_DIM
#define MAX_DIM 2 // Can be modified to a higher value
#endif

#ifndef _TENSORS_HEADER
#define _TENSORS_HEADER 


// DEFINITION OF TENSOR
struct _tensor
{
	int32_t shape[MAX_DIM];
	int32_t ndim;
	int32_t total_references; // Total references in stack
	int32_t fd; // file descriptor if the tensor is loaded on mmap
	off_t data_offset; // Only if the tensor is loaded on mmap
	bool on_mmap; // Flag to indicate whether the tensor is loaded on mmap or not. will be useful when i have to create/delete the tensor

	float * data; // Actual data (if any). Created either with a malloc ("standard" way) or with a mmap (loading from files)

	void * mmap_base;
	size_t mmap_size; // Incase mmap is used, store necessary data here (such as size et cetera)
	
	size_t max_capacity; // Maximum capacity of float array. Used ONLY to parse tensors from programs
	size_t current_capacity; // Also only for parsing tensors from programs
};


struct on_disk_tensor 
{
	int32_t shape[MAX_DIM];
	int32_t ndim; 
	off_t data_offset;
};

typedef struct _tensor * tensor; // Syntactic sugar
typedef struct on_disk_tensor tensor_disk_header;

//usefull stuff
bool check_is_shape_array(tensor shape_array); // DONE

// SOME OPERATIONS

// Part 1: Arithmetic
typedef float (*binary_op)(float, float); // Defines a generic operation between two floats, such as sum, subtraction, multiplication...- Also works when the output/inputs are booleans

tensor tensors_op_binary(tensor t1, tensor t2, binary_op op);	//DONE

tensor sum_tensors(tensor t1, tensor t2); 				//DONE
tensor diff_tensors(tensor t1, tensor t2);				//DONE
tensor mul_tensors_pointwise(tensor t1, tensor t2);		//DONE

// Part 2: COmparations
tensor cmp_less_tensors(tensor t1, tensor t2);	//DONE
tensor cmp_more_tensors(tensor t1, tensor t2);	//DONE
tensor cmp_eq_tensors(tensor t1, tensor t2);	//DONE

// Part 3: Logic operators
tensor tensors_op_binary_logical(tensor t1, tensor t2, binary_op op); //DONE

tensor or_tensors(tensor t1, tensor t2); // DONE
tensor and_tensors(tensor t1, tensor t2); //DONE

typedef float (*unary_op)(float); // Defines a generic operation on float, like f(x)=y
tensor tensor_unary_op(tensor t, unary_op op); // DONE
tensor not_tensor(tensor t); // DONE

// Part 4: Selection
tensor select_a_or_b(tensor t, tensor a, tensor b); // DONE

// tutte le parti 1, 2, 3, 4 sono gratuite e facilmente parallelizzabili
// anche le parti 6, 8 e 9


// Part 5: SOme specific operations	
tensor product_tensors(tensor t1, tensor t2);	//DONE
tensor dot_product_tensors(tensor t1, tensor t2); // DONE
tensor conv_tensors(tensor t, tensor k); // DONE

// Part 6: Operations on shape
tensor reshape_tensor(tensor t, tensor shape_array); // DONE
tensor flatten_tensor(tensor t); // DONE
tensor get_tensor_shape(tensor t); // DONE

// Part 7: Useful stuff
void print_tensor(tensor t); 			// DONE
tensor rand_tensor(tensor shape_array); // DONE

// Part 8: Other pointwise operators
tensor relu_tensor(tensor t); //DONE
tensor min_tensor(tensor t1, tensor t2); //DONE
tensor max_tensor(tensor t1, tensor t2); // DONE

// Part 9: Reductions
tensor reduct_sum_tensor(tensor t); //DONE

// Part 10: fillings
tensor fill_shape_value(tensor shape_array, tensor v); // DONE

// ASSUMPTIONS FOR PARTS 1, 2, 3, 4: TENSORS MUST HAVE THE **SAME** SIZE!!!!!!
// ASSSUMPTIONS FOR PART 3 AND 4: TENSORS MUST HAVE VALUES EITHER 0 OR 1 
// ASSUMPTIONS FOR PART 5: THE SHAPES MUST MATCH, CONVOLUTION RETURNS SAME SHAPE AS t, DOT PRODUCT MUST HAVE 1D TENSORS

// part 10. my functions
int free_tensor(tensor t); // Frees tensor from memory, according to whether it's an allocated memory block in heap or as a shared memory zone with mmap

#endif