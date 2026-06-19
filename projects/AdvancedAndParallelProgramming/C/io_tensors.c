// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    18.05.2026


// THIS FILE IMPLEMENTS THE I/O OPERATIONS
#include "tensors.h"
#include "io_tensors.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

// Assume P5 binary variant for .pgm files
tensor read_pgm(char * filename)
{
    //  Reads a PGM file and returns the resulting tensor.
    //  Input: char* file name, the filename. Output: tensor t, resulting tensor. 
    //  N.B. This assumes P5 PGM format (values from 0 to 255) and no comments in the .pgm file

	FILE* fp = fopen(filename, "rb");
	if(fp == NULL) { perror("Error in opening file"); return NULL; } // Error: file not readable

	char magic[3]; // PGM format specifier
	if(fscanf(fp, "%2s", magic)!=1){ fclose(fp); printf("PGM File error\n"); return NULL;}

	if(strcmp(magic, "P5") != 0){ printf("Unsupported PGM file (must be PGM5)\n"); fclose(fp); return NULL;}

	int width; int height; int maxval;

	if(fscanf(fp, "%d %d", &width, &height) != 2){ printf("Error in reading PGM file (probably corrupted)\n"); fclose(fp); return NULL; }

	int total = width*height;
	if(fscanf(fp, "%d", &maxval) != 1){ printf("Error in reading PGM file (probably corrupted)\n"); fclose(fp); return NULL; }

	if(maxval>255){ printf("Error: Unsupported PGM file (only 8bit, maxval is higher)\n"); fclose(fp); return NULL; }

	// initialize the tensor and then start consuming pixels one by one
	tensor t = malloc(sizeof(*t));

	t->ndim = 2;
	t->shape[1] = width;
	t->shape[0] = height;
	t->on_mmap=0;

	t->data = malloc(sizeof(float)*total);

	// consume pixels and store in the data arrayù
	fgetc(fp); // consume one whitespace

	// uhh ill trust casting process
	// NO PARALLELIZATION SINCE FREAD IS A SEQUENTIAL OPERATION
	for(int i=0; i<total; i++)
	{
		uint8_t pixel; // pixel of PGM is an unsigned 8 bit integer
		if(fread(&pixel, sizeof(uint8_t), 1, fp)!=1){ printf("ERROR: Unexpected EOF of pgm file. returning mnull"); free(t->data); free(t); return NULL; }
		t->data[i] = (float)pixel;
	}

	fclose(fp);

	// Last step: normalize values in [0,1]
    // Not parallelized, performances keep degrading (tried images up to 10k x 10k)
	for(int i=0; i<total; i++)
	{
		t->data[i] = (t->data[i])/255;
	}
	return t;
}

void write_pgm(tensor t, char * filename)
{
    //  Writes tensor t as a pgm file with name filename.
    //  Input: char* filename, the filename. tensor t, the tensor. 
    //  Ouput: None, returns void
    //  N.B. This writes in P5 format, so from 0 to 255

	// Step -1: OPen file
	FILE* fp = fopen(filename, "wb");
	if(!fp){ perror("Error opening file"); return; }

	// Step 0: Check 2D array, as well as handle invalid values (outside [0, 1])
	if(t->ndim != 2){ printf("ERROR: INVALID ARRAY SHAPE. TERMINATING FUNCTION\n"); fclose(fp); return; }

	// normalize values in [0, 255] and clip out of bound values
	int total = (t->shape[0])*(t->shape[1]);
	tensor t_write = malloc(sizeof(*t_write));
	t_write -> ndim = 2;
	t_write->shape[0] = t->shape[0];
	t_write->shape[1] = t->shape[1];
	t_write->data = malloc(sizeof(float)*total);
    
    // Not parallelized, performances keep degrading (tried images up to 10k x 10k)
	for(int i = 0; i<total; i++)
	{
		// No branchless programming since it's (kinda) expected that values are in the correct range)
		if(t->data[i] < 0)
		{
			t_write->data[i] = 0;
		}

		if(t->data[i] > 1)
		{
			t_write->data[i] = 255;
		}

		else t_write->data[i] = 255 * (t->data[i]);
	}

	// start writing!!!
	fprintf(fp, "P5\n");
	fprintf(fp, "%d %d\n", t->shape[1], t->shape[0]);
	fprintf(fp, "255\n");

	for(int i=0; i<total; i++)
	{
		fputc((uint8_t)t_write->data[i], fp);
	}

    fclose(fp);

	free(t_write->data); free(t_write); // end by removing trash
}

tensor read_file_mmap(char *filename)
{
    //  Reads a binary file and returns the resulting tensor.
    //  Input: char* file name, the filename. Output: tensor t, resulting tensor. 
    //  N.B. This assumes file format specified in the project guidelines.

	// check some preliminary stuff
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Error in opening file");
        return NULL;
    }

    tensor_disk_header h = {0};

    if (fread(&h, sizeof(h), 1, fp) != 1) {
        perror("Error in reading file");
        fclose(fp);
        return NULL;
    }

    if (h.ndim < 1 || h.ndim > MAX_DIM) {
        printf("ERROR: invalid tensor number of dimensions\n");
        fclose(fp);
        return NULL;
    }

    size_t total = 1;

    for (int d = 0; d < h.ndim; d++) {
        if (h.shape[d] <= 0) {
            printf("ERROR: invalid shape\n");
            fclose(fp);
            return NULL;
        }

        total *= (size_t)h.shape[d];
    }

    size_t data_size = sizeof(float) * total;
    size_t map_size = (size_t)h.data_offset + data_size;

    tensor t = malloc(sizeof(*t));

    t->ndim = h.ndim;

    for (int i = 0; i < MAX_DIM; i++) {
        t->shape[i] = h.shape[i];
    }

    t->on_mmap = 1;
    t->mmap_size = map_size;

    void *base = mmap(
        NULL,
        map_size,
        PROT_READ,
        MAP_PRIVATE,
        fileno(fp),
        0
    );

    if (base == MAP_FAILED) {
        perror("Error in mmap");
        fclose(fp);
        free(t);
        return NULL;
    }

    t->mmap_base = base;
    t->data = (float *)((char *)base + h.data_offset); // map the data from the data offset onwards

    fclose(fp);
    return t;
}

void write_tensor(tensor t, char *filename)
{
    //  Writes tensor t as a binary file with name filename.
    //  Input: char* file name, the filename. tensor t, the tensor. 
    //  Ouput: None, returns void
    //  N.B. This writes in the format specified in the project guidelines

    if (t->ndim < 1 || t->ndim > MAX_DIM) {
        printf("ERROR: invalid ndim\n");
        return;
    }

    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        perror("Error in opening file");
        return;
    }

    tensor_disk_header h = {0};

    h.ndim = t->ndim;

    for (int i = 0; i < MAX_DIM; i++) {
        if (i < t->ndim)
            h.shape[i] = t->shape[i];
        else
            h.shape[i] = 0;
    }

    h.data_offset = DATA_ALIGNMENT;

    if (sizeof(h) > (size_t)h.data_offset) {
        printf("ERROR: header size larger than offset\n");
        fclose(fp);
        return;
    }

    if (fwrite(&h, sizeof(h), 1, fp) != 1) {
        perror("Error in writing header");
        fclose(fp);
        return;
    }

    size_t padding_size = (size_t)h.data_offset - sizeof(h);

    char padding[DATA_ALIGNMENT] = {0}; // all zeros

    if (fwrite(padding, 1, padding_size, fp) != padding_size) {
        perror("Error in writing padding");
        fclose(fp);
        return;
    }

    size_t total = 1;

    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] <= 0) {
            printf("ERROR: invalid shape\n");
            fclose(fp);
            return;
        }

        total *= (size_t)t->shape[i];
    }

    size_t written = fwrite(t->data, sizeof(float), total, fp);

    if (written != total) {
        printf("Written and expected floats do not match");

        if (ferror(fp)) {
            perror("error in fwrite:");
        }

        fclose(fp);
        return;
    }

    fclose(fp);
}