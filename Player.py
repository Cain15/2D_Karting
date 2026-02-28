import pygame

class Player:
    def __init__(self, start_pos):
        self.player_pos = start_pos
        self.player_angle = 90
        self.player = pygame.image.load("assets/player.png")
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