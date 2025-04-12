import os
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt


class LevelImageGen:
    """
    Generates level images from ASCII maps for Sonic levels.
    """

    def __init__(self, sprite_path:str):
        """
        Initializes the LevelImageGen object and loads the necessary sprites.

        Args:
            sprite_path (str): Path to the folder containing sprite sheets.
        """
        self.sprite_dict = self.load_sprites(sprite_path)

    def load_sprites(self, sprite_path:str):
        """
        Loads and crops all necessary sprites for the Sonic level generation.

        Args:
            sprite_path (str): Path to the folder containing sprite sheets.

        Returns:
            dict: A dictionary of loaded and resized sprites keyed by their names.
        """
        with open(os.path.join(sprite_path, r"level_1_ascii_to_image_dict.json"), "r", encoding="utf-8") as f:
            sprite_dict = json.load(f)

        return sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """
        Prepares the sprite and the box for pasting in the level image based on the ASCII map.

        Args:
            ascii_level (list): 2D list representing the ASCII map of the level.
            sprite_key (str): Key that determines which sprite to use.
            curr_x (int): The current x-coordinate of the tile in the ASCII map.
            curr_y (int): The current y-coordinate of the tile in the ASCII map.

        Returns:
            tuple: A tuple containing the sprite (PIL Image) and the bounding box (tuple).
        """

        def has_three_ground_blocks_below(ascii_level, curr_x, curr_y):
            """
            Checks if there are at least 3 ground blocks directly below the given position.
            Args:
                ascii_level: 2D list of the ASCII level.
                curr_x: X-coordinate of the current position.
                curr_y: Y-coordinate of the current position.

            Returns:
                bool: True if there are at least 3 ground blocks directly below, False otherwise.
            """

        box = ()
        sprite = None
        b_multiplier = 64

        if sprite_key not in self.sprite_dict.keys():
            return sprite, box
        
        sprite_data = np.array(self.sprite_dict[sprite_key], dtype=np.uint8)
        print("Sprite data shape:", sprite_data.shape)  # Debugging
        sprite = Image.fromarray(sprite_data, mode="L")
        sprite = sprite.resize((b_multiplier, b_multiplier))

        plt.imshow(sprite, cmap="gray")
        plt.title(f"Sprite: {sprite_key}")
        plt.show()


        box = (
            curr_x * b_multiplier,
            curr_y * b_multiplier,
            (curr_x + 1) * b_multiplier,
            (curr_y + 1) * b_multiplier,
        )

        return sprite, box

    def render(self, ascii_level):
        """
        Generates the full level image from the ASCII map.

        Args:
            ascii_level (list): 2D list representing the ASCII map of the level.

        Returns:
            Image: The generated level image as a PIL Image.
        """
        b_multiplier = 64
        len_level = len(ascii_level[0])
        height_level = len(ascii_level)
        dst = Image.new("L", (len_level * b_multiplier, height_level * b_multiplier))

        # Render background first
        background = np.array(self.sprite_dict["Ƣ"]).astype(np.uint8)
        background = Image.fromarray(background, mode="L").resize(
            (b_multiplier, b_multiplier)
        )
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(
                    background,
                    (
                        x * b_multiplier,
                        y * b_multiplier,
                        (x + 1) * b_multiplier,
                        (y + 1) * b_multiplier,
                    ),
                )

        # Render each sprite based on ASCII map
        for y, row in enumerate(ascii_level):
            for x, char in enumerate(row):
                sprite_key = char
                if sprite_key:
                    sprite, box = self.prepare_sprite_and_box(
                        ascii_level, sprite_key, x, y
                    )
                    if sprite and box:
                        dst.paste(
                            sprite, box, mask=sprite
                        )  # Use sprite as its own mask

        return dst
