// NOME	COGNOME	MATRICOLA   MATRICOLA	DATA
// DINO	MENG	SM3201466	20241265    16.05.2026

// This is the header file that handles I/O operations on tensors

#include "tensors.h"
#include <stdint.h>
#include <string.h>

#define DATA_ALIGNMENT 64 // data alignment for when i'm writing tensors to file

#ifndef _IO_TENSORS
#define _IO_TENSORS 

tensor read_pgm(char * filename);
void write_pgm(tensor t, char * filename);
tensor read_file_mmap(char * filename);
void write_tensor(tensor t, char * filename);

#endif 