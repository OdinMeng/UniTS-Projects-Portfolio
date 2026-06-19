"""
======
NOME
    DINO
COGNOME
    MENG
MATRICOLA
    SM3201466
    20241265
======

This code implements the VirtualVRAM class, which allows to load tile/sprite sheets in binary format.
Basically decodes the respective files in a matrix of palette indexes.
"""

import numpy as np 
from numpy.typing import NDArray

class VirtualVRAM:
    def __init__(self, filepath: str, width: int = 256, height: int = 256):
        self.filepath : str = filepath 
        self.width : int = width
        self.height : int = height
        self.buffer : NDArray[np.uint8]
        self.processed = False
        
    def process_file(self) -> None:
        """
        This file simply does what the class is supposed to do, i.e. obtain the image buffer of the binary file. More specifically,
        1. Initializes a zero image buffer 
        2. Loads the binary file into an array, validates the shapes 
        3. For each byte of the number in the binary array, calculate its low and high mask
        4. Fill the image buffer in alternating order

        Input: None, provided in the class
        Output: A NDArray with unsigned 8-bit integers 
        """
        image_buffer = np.zeros((self.width * self.height), dtype=np.uint8)

        try:
            binary_array = np.fromfile(self.filepath, dtype=np.uint8)

        except Exception as e: # Not the most descriptive thing, but usually is due to I/O errors (e.g. file does not exist)
            raise RuntimeError(f"Failed to process binary file {self.filepath}") from e
        
        # Validate shapes
        if(binary_array.shape[0]*2 != self.width*self.height):
            raise ValueError(f"Binary file provided has invalid size: must be exactly {self.width*self.height/2} bytes, while the provided file contains {binary_array.shape[0]} bytes")

        # Calculate the masks of the binary array
        low_mask = (binary_array & 0b00001111) # Lower part of each byte
        high_mask = ((binary_array & 0b11110000) >> 4) # Higher part of each byte: ofc we shifted to right by 4 bits

        image_buffer[::2] = high_mask # Even: the higher bigs
        image_buffer[1::2] = low_mask

        image_buffer = image_buffer.reshape((self.width, self.height))
        self.processed = True
        self.buffer = image_buffer 

    def get_buffer(self):
        if not self.processed:
            raise RuntimeError("Must run .process_file() first")
        return self.buffer 
    