#include <stdio.h>
#include "io_tensors.h"
#include "tensors.h"
#include <stdlib.h>

int main(void)
{
	/*
	tensor t = read_pgm("./examples/cray-2.pgm");

	if(t==NULL){ printf("sike\n");}

	printf("%d %d\n", t->shape[0], t->shape[1]);
	printf("x[0] = %f\n", t->data[0]);

	float data_1[] = {
		-1, -1, -1,
		0, 1, 0,
		-1, -1, -1
	};


	tensor t1 = malloc(sizeof(*t1));
	t1 -> ndim = 2;
	t1 -> shape[0] = 3;
	t1 -> shape[1] = 3;
	t1 -> data = data_1;

	tensor t_new = conv_tensors(t, t1);

	write_pgm(t_new, "./new_pgm.pgm");
	write_tensor(t_new, "./file.bin");

	tensor t_fake = read_file_mmap("./file.bin");
	*/

	/*
	tensor t = malloc(sizeof(*t));
	t->ndim = 2;
	t->shape[0] = 50;
	t->shape[1] = 50;
	t->data = malloc(sizeof(float)*1000*1000);
	for(int i=0; i<50*50; i++) t->data[i] = 1;

	write_tensor(t, "./file.bin");

	free(t->data); free(t);
	*/
	
	tensor t_read = read_file_mmap("./file.bin");
	t_read -> total_references = 0;
	printf("%d %d\n", t_read->shape[0], t_read->shape[1]);

	free_tensor(t_read);
}
