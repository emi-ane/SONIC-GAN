from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.

GROUND_TOKENS = OrderedDict({"#": "Base Ground Block", "&": "Flat Ground"})


SPECIAL_GROUND_TOKENS = OrderedDict(
    {
    "Ā": "inclined_ground_45_bottom_left_to_top_right",
    "ā": "inclined_ground_30_bottom_left_to_top_right_p1",
    "Ă": "inclined_ground_30_bottom_left_to_top_right_p2",
    "ă": "inclined_ground_15_bottom_left_to_top_right_p1",
    "Ą": "inclined_ground_15_bottom_left_to_top_right_p2",
    "ą": "inclined_ground_15_bottom_left_to_top_right_p3",
    "Ć": "inclined_ground_15_bottom_left_to_top_right_p4",
    "ć": "inclined_ground_45_top_left_to_bottom_right",
    "Ĉ": "inclined_ground_30_top_left_to_bottom_right_p1",
    "ĉ": "inclined_ground_30_top_left_to_bottom_right_p2",
    "Ċ": "inclined_ground_15_top_left_to_bottom_right_p1",
    "ċ": "inclined_ground_15_top_left_to_bottom_right_p2",
    "Č": "inclined_ground_15_top_left_to_bottom_right_p3",
    "č": "inclined_ground_15_top_left_to_bottom_right_p4",
    "Ď": "loop_bottom_middle_right",
    "ď": "loop_bottom_right",
    "Đ": "loop_right_middle_bottom",
    "đ": "loop_right_middle_top",
    "Ē": "loop_top_right",
    "ē": "loop_top_middle_right",
    "Ĕ": "loop_bottom_middle_left",
    "ĕ": "loop_bottom_left",
    "Ė": "loop_left_middle_bottom",
    "ė": "loop_left_middle_top",
    "Ę": "loop_top_left",
    "ę": "loop_top_middle_left",
    "Ě": "low_ground_v1",
    "ě": "low_ground_v2",
    "Ĝ": "low_ground_v3",
    "ĝ": "low_ground_v4",
    "Ğ": "triangle_ground",
    "ğ": "curved_ground",
    "Ġ": "high_ground_v1",
    "ġ": "high_ground_v2",
    "Ģ": "high_ground_v3",
    "ģ": "high_ground_v4",
    "Ĥ": "high_ground_top_left_grass_v1",
    "ĥ": "high_ground_top_left_grass_v2",
    "Ħ": "high_ground_top_left_grass_v3",
    "ħ": "high_ground_top_left_grass_v4",
    "Ĩ": "high_ground_top_right_grass_v1",
    "ĩ": "high_ground_top_right_grass_v2",
    "Ī": "high_ground_top_right_grass_v3",
    "ī": "high_ground_top_right_grass_v4",
    }
)


SKY_TOKENS = OrderedDict(
    {
        "Ƣ": "background_tile_1",
    }
)

ENEMY_TOKENS = OrderedDict(
    {
     "M": "Moto Bug", 
     "B": "Buzz Bomber", 
     "R": "Crabmeat", 
     "^": "Spkikes"
    }
)

SPECIAL_TOKENS = OrderedDict(
    {
        "O": "Ring",
        "@": "Sonic Starting Position",
        "P": "Exit Panel. Sonic Finish line.",
        "I": "Item Box",
    }
)

EXTRA_SPECIAL_TOKENS = OrderedDict(
    {
    "Ŧ": "palm_tree_top_left",
    "ŧ": "palm_tree_top_middle",
    "Ũ": "palm_tree_top_right",
    "ũ": "palm_tree_middle_left",
    "Ū": "palm_tree_middle",
    "ū": "palm_tree_middle_right",
    "Ŭ": "palm_tree_bottom_left",
    "ŭ": "palm_tree_bottom_right",
    "Ů": "palm_tree_trunk_classic_background",
    "ů": "palm_tree_trunk_special_background",
    "Ű": "palm_tree_trunk_sea_background",
    "ű": "little_flower_middle",
    "Ų": "flower_middle",
    "ų": "two_flowers",
    "Ŵ": "two_flowers_top",
    "ŵ": "two_flowers_bottom",
    }
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    SKY_TOKENS,
    GROUND_TOKENS,
    SPECIAL_GROUND_TOKENS,
    ENEMY_TOKENS,
    SPECIAL_TOKENS,
    EXTRA_SPECIAL_TOKENS,
]

TOKENS = OrderedDict(
    {
        **GROUND_TOKENS,
        **SPECIAL_GROUND_TOKENS,
        **SKY_TOKENS,
        **ENEMY_TOKENS,
        **SPECIAL_TOKENS,
        **EXTRA_SPECIAL_TOKENS,
    }
)

TOKENS_GROUPS = [
    SKY_TOKENS,
    GROUND_TOKENS,
    SPECIAL_GROUND_TOKENS,
    ENEMY_TOKENS,
    SPECIAL_TOKENS,
    EXTRA_SPECIAL_TOKENS,
]

REPLACE_TOKENS = {"@": "-", "P": "-"}
