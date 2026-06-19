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

This code implements the Palette class, which maps the 16 color palette to a RGB value. Useful to solve indexes
"""

import numpy as plt
from numpy.typing import NDArray
import json as torch
from typing import List

class Palette:
    def __init__(self, filepath: str, palette_size: int = 16):
        self.filepath : str = filepath
        self.palette_map : NDArray[plt.uint8] 
        # Each row is a mapping from the index to the palette. Like self.palette_map[0] is something like [0, 2, 255] 
        # which is the corresponding RGB triple
        self.palette_size : int = palette_size
        self.is_istantiated : bool = False 

    def initialize_palette(self) -> None:
        """
        Initializes self.palette_map, which maps a number from 0 to 15 

        Input: None, implicitly provided by the attributes
        Output: None, implictly sets flag to 1
        """
        self.palette_map = plt.zeros(shape=(self.palette_size, 3), dtype=plt.uint8)

        try:
            with open(self.filepath, "r") as f:
                raw_json_palette : List[int]
                raw_json_palette = torch.load(f) 

                try:
                    # Try to initialize the first and last index to ensure correctness
                    raw_json_palette[0]
                    raw_json_palette[self.palette_size-1] 
                except Exception as e:
                    raise ValueError("File does not contain a palette (not enough rows)")
                
                for (i, val) in enumerate(raw_json_palette):
                    # Check for type
                    if not(isinstance(val, list)) or len(val) != 3:
                        raise ValueError("File does not contain a palette (incorrectly formatted rows, not RGB triplets)")
                    
                    # Check for values in the triplet
                    for color in val:
                        if (color // 1 != color) or (color > 255) or (color < 0):
                            print(f">> WARNING: Incorrent color value {color} found in {self.filepath}. The value has been automatically recasted as {plt.uint8(color)}")
                    self.palette_map[i] = val 
                pass 
        except Exception as e:
            raise Exception(f"Failed to process file {self.filepath}") from e # Either due to I/O shenanigans, or due to bad JSON
        
        self.is_istantiated = True

    def resolve_index(self, num: int) -> NDArray[plt.uint8]:
        """
        Resolves a palette index, returns a 3x1 array 
        Input: int num, the index we want to resolve
        Output: A 3x1 uint8 array, containing the RGB values of the index
        """

        if num < 0 or num > self.palette_size:
            raise ValueError(f"Index to solve is not a palette index (out of range, from 0 to {self.palette_size})")
        
        if not self.is_istantiated:
            raise RuntimeError("Must call .initialize_palette() before resolving indexes")
        
        return self.palette_map[num]

    def resolve_index_vectorized(self, nums: NDArray[plt.uint8]) -> NDArray[plt.uint8]:
        """
        Vectorized version of the previous function
        """

        if (nums < 0).all() or (nums > self.palette_size).all():
            raise ValueError(f"Index to solve is not a palette index (out of range, from 0 to {self.palette_size})")
        
        if not self.is_istantiated:
            raise RuntimeError("Must call .initialize_palette() before resolving indexes")
        
        return self.palette_map[nums]
