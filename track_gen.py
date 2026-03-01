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

    while True:
        order.append(current)
        visited.add(current)

        if current == start:
            neighbors = [(current[0]-1, current[1])]
        else:
            neighbors = [(current[0]-1, current[1]), (current[0]+1, current[1]), (current[0], current[1]-1), (current[0], current[1]+1)]
            i = 0
            while i < len(neighbors):
                if track[neighbors[i][1]][neighbors[i][0]] == Tile.GRASS:
                    neighbors.remove(neighbors[i])
                    i = i - 1
                i += 1

        # remove the tile we came from
        if prev in neighbors:
            neighbors.remove(prev)

        if not neighbors:
            break

        next_tile = neighbors[0]
        prev = current
        current = next_tile

        if current == start:
            break

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
            wx = tx * TILE_SIZE + TILE_SIZE / 2
            wy = ty * TILE_SIZE + TILE_SIZE / 2
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
    if player.prev_corner_distance:
        progress = player.prev_corner_distance - distance
        reward += progress * 0.05
    player.prev_corner_distance = distance

    # Waypoint reached
    if distance < 40:
        reward += 3.0
        player.current_waypoint_index = (player.current_waypoint_index + 1) % len(waypoints)
        player.prev_corner_distance = float("inf")

    return reward

