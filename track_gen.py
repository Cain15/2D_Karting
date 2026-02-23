import enum

TILE_SIZE = 64
tiles_x = int(1280 / TILE_SIZE)
tiles_y = int(768 / TILE_SIZE)

"""
    0: grass
    1: straight up
    2: straight down
    3: straight left
    4: straight right
    5: corner down to right
    6: corner down to left
    7: corner up to right
    8: corner up to left
"""

class Tile(enum.Enum):
    GRASS = 0
    STRAIGHT_UP = 1
    STRAIGHT_DOWN = 2
    STRAIGHT_LEFT = 3
    STRAIGHT_RIGHT = 4
    CORNER_DOWN_RIGHT = 5
    CORNER_DOWN_LEFT = 6
    CORNER_UP_RIGHT = 7
    CORNER_UP_LEFT = 8

# with open("track1.tr", "w") as f:
#     for j in range(tiles_y):
#         line = "0" * tiles_x
#         line += "\n"
#         f.write(line)

def read_track(file):
    track = []
    with open(file, "r") as f:
        for line in f:
            line = line.replace(" ", "").strip("\n")
            row = []
            for tile in line:
                row.append(Tile(int(tile)))
            track.append(row)
    return track

read_track("track1.tr")
