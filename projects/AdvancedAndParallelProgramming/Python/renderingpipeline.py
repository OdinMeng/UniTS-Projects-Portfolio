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

Implements the RenderingPipeline class, which basically runs the entire rendering pipeline, returning the final frame buffer
indexed with RGB colours and saves the matrix as a .png file with pillow
"""

from virtualvram import VirtualVRAM
from palette import Palette
from blitter import Blitter
from sceneparser import SceneParser
import numpy as pd
from PIL import Image

#NOTE: Interacts with every other class, of course. Can be thought as the orchestrator of the components
#NOTE: Here each part do not have to be intialized (can be done, but not strictly necessary) as the pipeline automatically does it
class RenderingPipeline:
    def __init__(self, tiles_path: str, sprites_path: str,
                palette_path: str, scene_path : str, output_path: str,
                output_width: int = 640, output_height: int = 480):
        self.tiles_path = tiles_path
        self.sprites_path = sprites_path
        self.palette_path = palette_path
        self.scene_path = scene_path
        self.output_width: int = output_width
        self.output_height: int = output_height
        self.output_path : str = output_path
        
    def run(self):
        """
        Runs the entire pipeline, from files parsing to final image output delivery. 
        """
        # Create necessary objects first, if existng
        try:
            self.vram_tiles = VirtualVRAM(self.tiles_path)
            self.vram_sprites = VirtualVRAM(self.sprites_path)
            self.palette = Palette(self.palette_path)
            self.scene_parser = SceneParser(self.scene_path)

            self.vram_tiles.process_file()
            self.vram_sprites.process_file()
            self.palette.initialize_palette()
            self.scene_parser.parse_scene()

            self.blitter = Blitter(self.vram_tiles, self.vram_sprites, self.palette)

        except Exception as e:
            raise RuntimeError("Failed initializing the necessary objects for the main pipeline. See the following exceptions.") from e
            
        # Validate some stuff across the workers
        if( self.scene_parser.parsed_scene['transparent_index'] < 0 or self.scene_parser.parsed_scene['transparent_index']  >= self.palette.palette_size):
            raise ValueError(f"Transparent index out of range, must be between [0 and {self.palette.palette_size}) but received index {self.scene_parser.parsed_scene['transparent_index'] }")

        # Start drawing with the blitter!
        canvas = pd.zeros((self.output_height, self.output_width, 3), dtype=pd.uint8)

        # Iterate the tiles
        for (y, row) in enumerate(self.scene_parser.parsed_scene['tile_map']):
            for (x, tileid) in enumerate(row):
                self.blitter.extract_draw_tile(canvas, tileid, x, y)

        # Now iterate with the sprites
        for kid in self.scene_parser.parsed_scene['sprites']:
            self.blitter.extract_draw_sprite(canvas, 
                                             kid['id'],
                                             kid['x'], kid['y'],
                                             kid['flip_h'], kid['flip_v'],
                                             kid['rotation'],
                                             self.scene_parser.parsed_scene['transparent_index'])

        # open image and save
        img = Image.fromarray(canvas)
        img.save(self.output_path)