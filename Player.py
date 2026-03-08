from pygame import Vector2


class Player:
    def __init__(self, start_pos: Vector2, player_id: int = 0):
        self.player_pos = start_pos.copy()
        self.player_angle = 90.0
        self.player_acceleration = 55
        self.player_deceleration = 108
        self.player_velocity = 0
        self.player_max_velocity = 240
        self.dsq = False
        self.prev_tile = None
        self.seen_finish = False
        self.tiles_visited = []
        self.amount_warnings = 0
        self.start_time = None
        self.prev_state = None
        self.prev_action = None
        self.current_waypoint_index = 0
        self.prev_corner_distance = float("inf")
        self.player_id = player_id
