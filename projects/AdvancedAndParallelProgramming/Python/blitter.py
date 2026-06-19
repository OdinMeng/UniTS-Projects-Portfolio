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

Implements the Blitter class, which extracts tiles and sprites from the respective sheets and apply group transformations on 
sprites.
Then copies the result on the frame buffer.
"""

from virtualvram import VirtualVRAM
from palette import Palette
import numpy as np 
from numpy.typing import NDArray

# NOTE: This interacts with any Palette and VirtualVRAM object. This is NECESSARY!
# Blitter is an extension of Palette and VirtualVRAM, so on instantiation I must insert both of them
class Blitter:
    def __init__(self, vram_tiles: VirtualVRAM, vram_sprites: VirtualVRAM, palette: Palette, 
                 tile_size : int = 32, sprite_size : int = 64):
        # IMPORTANT NOTE: THIS ASSUMES THAT ALL OF THE OBJECTS BELOW ARE PROPERLY ISTANTIATED (if required)
        self.vram_tiles : VirtualVRAM = vram_tiles
        self.vram_sprites : VirtualVRAM = vram_sprites
        self.palette : Palette = palette 

        # Some useful constants, although for the scopes of this project they're automatically set to default values
        self.tile_size = tile_size
        self.sprite_size = sprite_size

    def clip_pixel_position(self, x: int, m: int, M: int) -> int:
        return min(max(x, m), M)

    def extract_draw_tile(self, frame_buffer : NDArray[np.uint8], tile_idx : int, tile_posx : int, tile_posy : int):
        """
        Extracts specified tile and draws it into the frame buffer.
        Input(s):
            frame_buffer: the frame buffer on which the blitter will draw on
            tile_idx: ID of the tile
            tile_posx, tile_posy: Initial position of the tile which marks the "start" of the drawn area, considered as a multiple of tile_size

        Output: None, directly modifier frame_buffer (inplace operator)
        """
        # 1: Get height and width of the buffer
        image_height = frame_buffer.shape[0]
        image_width = frame_buffer.shape[1]

        buffer = self.vram_tiles.get_buffer()

        # 2: Convert tile index to its actual position on the buffer. Clip if needed (to avoid out-of-bound situations)
        tile_idx_y, tile_idx_x = divmod(tile_idx, 8)

        x1 = self.clip_pixel_position(tile_idx_x * self.tile_size, 0, self.vram_tiles.width)
        x2 = self.clip_pixel_position((tile_idx_x + 1) * self.tile_size, 0, self.vram_tiles.width)
        y1 = self.clip_pixel_position(tile_idx_y * self.tile_size, 0, self.vram_tiles.height)
        y2 = self.clip_pixel_position((tile_idx_y + 1) * self.tile_size, 0, self.vram_tiles.height)

        # 3: Get crop destination
        x1_ = self.clip_pixel_position(tile_posx * self.tile_size, 0, image_width)
        x2_ = self.clip_pixel_position((tile_posx + 1) * self.tile_size, 0, image_width)
        y1_ = self.clip_pixel_position(tile_posy * self.tile_size, 0, image_height)
        y2_ = self.clip_pixel_position((tile_posy + 1) * self.tile_size, 0, image_height)

        # 4: Get the minumum lengths/heights to determine if cropping is needed
        dw = min(x2 - x1, x2_ - x1_)
        dh = min(y2 - y1, y2_ - y1_)

        if dw <= 0 or dh <= 0:
            return

        # 5: Solve and paste
        selected = buffer[y1:y1 + dh, x1:x1 + dw]
        selected_solved = self.palette.resolve_index_vectorized(selected)

        frame_buffer[y1_:y1_ + dh, x1_:x1_ + dw] = selected_solved

    def extract_draw_sprite(self, frame_buffer : NDArray[np.uint8], sprite_idx : int, sprite_posx : int, sprite_posy : int,
                            flip_h: bool, flip_v: bool, rotate: int, transparent_id: int):
        """
        Same thing as extract_draw_tile but with sprites isntead of tiles..
        Input(s):
            frame_buffer: the frame buffer on which the blitter will draw on
            sprite_idx: ID of the sprite
            sprite_posx, sprite_posy: Initial position of the tile which marks the "start" of the drawn area
            flip_h, flip_v, rotate: Values to indicate which (and how) transformations belonging in the dihedral group D4  
            transparent_id: Colour ID of the transparent colour, which will be not considered for the frame_buffer

        Output: None, directly modifier frame_buffer (inplace operator)

        """
        # Scene parser feeds tile_idx, tile_posx and tile_posy
        # Starts drawing the tile on the frame buffer, ignoring eventualy out-of-bound cases. 
        
        image_width = frame_buffer.shape[1]
        image_height = frame_buffer.shape[0]

        # Step 0: Extract the tile from the VRAM tile image buffer
        buffer = self.vram_sprites.get_buffer()

        # Step 1: select boxes of the frame buffers
        # imagine tiles as a matrix in row-major order, so we have like something like [0, 1, 2] [3, 4, 5] [6, 7, 8]
        # so we do the eucliean division of the index by 4 (since each sprite is a "4x4 matrix")
        sprite_idx_y, sprite_idx_x = divmod(sprite_idx, 4)

        x1 = self.clip_pixel_position(sprite_idx_x*self.sprite_size, 0, self.vram_sprites.width)
        x2 = self.clip_pixel_position(sprite_idx_x*self.sprite_size+self.sprite_size, 0, self.vram_sprites.width)
        y1 = self.clip_pixel_position(sprite_idx_y*self.sprite_size, 0, self.vram_sprites.height) 
        y2 = self.clip_pixel_position(sprite_idx_y*self.sprite_size+self.sprite_size, 0, self.vram_sprites.height)
        selected = buffer[y1: y2, x1: x2]
        selection_mask = (selected != transparent_id) # For transparent colours

        # Intermediate step: apply group transformations before resolving
        n_rotations = rotate // 90
        selected = np.rot90(selected, n_rotations)

        if flip_h:
            selected = selected[:, ::-1]

        if flip_v:
            selected = selected[::-1, :]

        sprite_w, sprite_h = selected.shape[:2] 

        # Calculate the paste destination
        x1_ = self.clip_pixel_position(sprite_posx, 0, image_width)
        x2_ = self.clip_pixel_position(sprite_posx+sprite_w, 0, image_width)
        y1_ = self.clip_pixel_position(sprite_posy, 0, image_height) 
        y2_ = self.clip_pixel_position(sprite_posy+sprite_h, 0, image_height)

        dw = x2_-x1_
        dh = y2_-y1_

        if dw <= 0 or dh <= 0:
            return # Nothing to copy

        # Recrop the copy zone if needed, might be necessary due to rotations
        selected = selected[
            y1_ - sprite_posy : y1_ - sprite_posy + dh,
            x1_ - sprite_posx : x1_ - sprite_posx + dw
        ]

        # Selection mask to avoid transparent colours
        selection_mask = (selected != transparent_id)

        # Step 1. Resolve each index 
        selected_solved = self.palette.resolve_index_vectorized(selected)

        # Step 2. Copy to the frame buffer
        dst = frame_buffer[y1_:y1_+dh, x1_:x1_+dw]
        dst[selection_mask] = selected_solved[selection_mask]
