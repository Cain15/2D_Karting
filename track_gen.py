import enum
import math

TILE_SIZE = 64
tiles_x = int(1280 / TILE_SIZE)
tiles_y = int(768 / TILE_SIZE)


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
    FINISH_LINE = 9


# all grass track generation

# with open("track1.tr", "w") as f:
#     for j in range(tiles_y):
#         line = "0" * tiles_x
#         line += "\n"
#         f.write(line)

def read_track(file: str):
    """
    Parse a .tr track file into a 2-D grid of Tile values.
    :param file: Path to the .tr track file.
    :return: A 2-D list where each element is a Tile enum value.
    """
    track = []
    with open(file, "r") as f:
        for line in f:
            line = line.replace(" ", "").strip("\n")
            row = []
            for tile in line:
                row.append(Tile(int(tile)))
            track.append(row)
    return track


def track_walk(track: list[list[Tile]], start: tuple[int, int]):
    """
    Walk the track starting from start and return tiles in traversal order.
    The walk follows a single-path traversal: at each step it moves to an
    unvisited, non-grass 4-directional neighbour, stopping when it returns
    to start (completing the loop) or has no valid neighbour.
    :param track: 2-D tile grid as returned by read_track.
    :param start: (x, y) tile coordinate to begin from.
    :return: Ordered list of (x, y) tile coordinates representing one full loop of the track.
    """
    visited = set()
    order = []
    current = start
    prev = None
    width = len(track[0])
    height = len(track)

    while True:
        order.append(current)
        visited.add(current)
        x, y = current
        # Generate all 4-directional neighbors
        candidates = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]
        neighbors = []
        for nx, ny in candidates:
            # Bounds check
            if not (0 <= nx < width and 0 <= ny < height):
                continue

            # Must not be grass
            if track[ny][nx] == Tile.GRASS:
                continue

            # Don't go back to where we just came from
            if (nx, ny) == prev:
                continue

            # Don't revisit tiles (except allow returning to start to close loop)
            if (nx, ny) in visited and (nx, ny) != start:
                continue

            neighbors.append((nx, ny))

        if not neighbors:
            break

        next_tile = neighbors[0]

        # If we returned to start, we're done
        if next_tile == start:
            break

        prev = current
        current = next_tile
    return order


class Waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def is_corner(tile: Tile):
    """
    Check if the given tile is a corner.
    :param tile: The tile we want checking
    :return: True if it is a corner, False otherwise
    """
    return tile in {
        Tile.CORNER_DOWN_RIGHT,
        Tile.CORNER_DOWN_LEFT,
        Tile.CORNER_UP_RIGHT,
        Tile.CORNER_UP_LEFT
    }


def generate_corner_waypoints(track: list[list[Tile]], ordered_tiles: list[tuple[int, int]]):
    """
    Build a list of waypoints at the centre of every corner tile, in lap order.
    Iterating through them in sequence guides an agent smoothly around the
    full circuit.
    :param track: 2D tile grid as returned by read_track.
    :param ordered_tiles: Tiles in traversal order, as returned by track_walk.
    :return: One Waypoint per corner tile, ordered by lap progression.
    """
    waypoints = []

    for (tx, ty) in ordered_tiles:
        tile = track[ty][tx]

        if is_corner(tile):
            # Convert tile position to world position (center of tile)
            wx = tx * 80 + 80 / 2
            wy = ty * 80 + 80 / 2
            waypoints.append(Waypoint(wx, wy))

    return waypoints
