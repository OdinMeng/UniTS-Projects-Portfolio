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

Main access point of the program. To be called as specified in the project specification. So something like 
python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>
"""

import sys 
from renderingpipeline import RenderingPipeline

if __name__ == "__main__":
    args = sys.argv

    # Define filepath strings
    palette_filepath : str
    scene_filepath : str
    tiles_filepath : str
    sprites_filepath : str
    output_filepath : str 

    # Load them if OK, otherwise raise exception
    if len(args) != 6:
        raise ValueError("Insufficient arguments: you need to specify the filepath to palette, scene, tiles, sprites" \
        "and the desired filepath of the output file.")
    
    else:
        palette_filepath = args[1]
        scene_filepath = args[2]
        tiles_filepath = args[3]
        sprites_filepath = args[4]
        output_filepath = args[5]

    try:
        pipeline = RenderingPipeline(tiles_filepath, sprites_filepath, palette_filepath, scene_filepath, output_filepath)
        pipeline.run()
    except Exception as e:
        raise RuntimeError("Exception occurred. Showing the exception stack") from e

    else:
        print(f"Pipeline ran successfully; see the output in {output_filepath}")

# To run test: python main.py ./examples/palette.json ./examples/scene.json ./examples/tiles.bin ./examples/sprites.bin test.png