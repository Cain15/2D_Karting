import enum
import math

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
    9: finish line
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
    FINISH_LINE = 9

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

def track_walk(track, start):
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

def is_corner(tile):
    return tile in {
            Tile.CORNER_DOWN_RIGHT,
            Tile.CORNER_DOWN_LEFT,
            Tile.CORNER_UP_RIGHT,
            Tile.CORNER_UP_LEFT
        }

def generate_corner_waypoints(track, ordered_tiles):
    waypoints = []

    for (tx, ty) in ordered_tiles:
        tile = track[ty][tx]

        if is_corner(tile):
            # Convert tile position to world position (center of tile)
            wx = tx * 80 + 80 / 2
            wy = ty * 80 + 80 / 2
            waypoints.append(Waypoint(wx, wy))

    return waypoints

def corner_reward(player, waypoints):
    if not waypoints:
        return 0

    reward = 0
    wp = waypoints[player.current_waypoint_index]

    dx = wp.x - player.player_pos.x
    dy = wp.y - player.player_pos.y
    distance = (dx**2 + dy**2) ** 0.5

    # Smooth reward for moving closer
    if not math.isinf(player.prev_corner_distance):
        progress = player.prev_corner_distance - distance
        reward += progress * 0.05
    player.prev_corner_distance = distance

    # Waypoint reached
    if distance < 55.57:
        # reward += 3.0
        player.current_waypoint_index = (player.current_waypoint_index + 1) % len(waypoints)
        player.prev_corner_distance = float("inf")

    return reward

