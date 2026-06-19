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

Implements the SceneParser class, which basically loads a JSON scene file and 
returns a complete descriptions on the background and sprites. 
"""

from typing import Dict, Any
import json as jason

class SceneParser:
    def __init__(self, filename : str):
        self.filename : str = filename
        self.parsed_scene : Dict[str, Any]
        self.is_intialized : bool = False

    def parse_scene(self):
        """
        Parses scene file with JSON. Also checks for contents well enough, ensuring almost that the required fields
        exist and they contain valid elements (e.g. transparent index must be an integer, et cetera...) in a shallow way

        Input: none, accesses internal variables
        Output: none, modifies internal variables
        """
        try:
            with open(self.filename, "r") as f:
                retval = jason.load(f)

                indexes_to_try = ['transparent_index',
                                  'tile_map',
                                  'sprites']
                
                expected_types : Dict[str, Any] = {'transparent_index': int,
                                  'tile_map': list,
                                  'sprites': list
                                  }
                
                expected_types_sprites : Dict[str, Any] = {
                    'id': int, 
                    'x': int, 'y': int, 
                    'flip_h': bool, 'flip_v': bool,
                    'rotation': int
                }

                # Checks that all of the fields exist or they are all right
                for index in indexes_to_try:
                    try:
                        retval[index]
                        if not(isinstance(retval[index], expected_types[index])):
                            raise ValueError(f"Index {index} is expecting type {expected_types[index]} but got type {type(retval[index])}")
                        
                        if index == 'tile_map':
                            # Check for tile map specifically: must be a 15x20 matrix with all integers inside
                            i: int = 0
                            for row in retval[index]:
                                j: int = 0
                                if not(isinstance(row, list)):
                                    raise ValueError(f"Index {index} is expecting a list of lists but didn't get a list of lists, instead got {type(row)}")
                                for elem in row:
                                    if not(isinstance(elem, int)):
                                        raise ValueError(f"Index {index} is expecting a list of lists with integers, but instead got with {type(elem)}")
                                    j += 1
                                i += 1

                                if j != 20:
                                    raise ValueError(f"Tile map expecting a matrix with 20 columns, instead got {j} columns")
                            if i!=15:
                                raise ValueError(f"Tile map expecting a matrix with 15 rows, instead got {i} rows")
                        
                        if index == 'sprites':
                            # Check for sprites specifially too: must contain their required fields and also check the rotation is a number in [0, 90, 180, 270]
                            for val in retval[index]:
                                if not(isinstance(val, dict)):
                                    raise ValueError(f"In sprites ")
                                for key in expected_types_sprites.keys():
                                    try: 
                                        val[key]
                                    except Exception as e:
                                        raise ValueError(f"Failure accessing key {key} of a sprite, see the following exception") from e
                                    
                                    if(not(isinstance(val[key], expected_types_sprites[key]))):
                                        raise ValueError(f"Key {key} of a sprite expected type {expected_types_sprites[key]} but instead got {type(val[key])}")
                                    
                                # Also check that rotations must be one of [0, 90, 180, 270]
                                    if key == 'rotation':
                                        if(val[key] not in [0, 90, 180, 270]):
                                            return Exception(f"Rotation must be either one of [0, 90, 180, 270], but got {val[key]}")

                    except Exception as e:
                        raise ValueError(f"Scene file {self.filename} is missing field {index}, or it exists but has invalid content. See the following exception") from e

        except Exception as e:
            raise RuntimeError("Error in running the scene parser.") from e
        
        self.is_intialized = True 
        self.parsed_scene = retval