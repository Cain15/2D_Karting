import pygame
import math
import random
from track_gen import read_track, Tile
import time

def get_tile_pos(pygame_pos):
    """
    Get the tile coordinates
    :param pygame_pos: The pygame coordinates
    :return: the tile coordinates
    """
    x = math.floor(pygame_pos.x / TILE_SIZE)
    y = math.floor(pygame_pos.y / TILE_SIZE)
    return x,y

# Screen size
screen_width = 1600
screen_height = 960

# Initialize window
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
running = True
dt = 0

# Tile variables
TILE_SIZE = 80
tiles_x = 20
tiles_y = 12
finish_tile = (16,10)

# Player properties
player_pos = pygame.Vector2(finish_tile[0]*TILE_SIZE, finish_tile[1]*TILE_SIZE + TILE_SIZE/2)
player_angle = 90
player = pygame.image.load("assets/player.png")
player = pygame.transform.scale(player, (20,40))
friction = 9
player_acceleration = 45 + friction
player_deceleration = 108 - friction
player_velocity = 0
player_max_velocity = 270

# Track generation
track = read_track('track1.tr')
corner = pygame.image.load('assets/corner.png')
corner = pygame.transform.scale(corner, (80,80))
straight = pygame.image.load('assets/straight.png')
straight = pygame.transform.scale(straight, (80,80))

# Progress tracking
seen_finish = False
prev_tile = None
min_visited_tiles = 68
tiles_visited = []
amount_warnings = 0

# Game state variables
pause = 0
message = ""
dsq = False

# Keep lap times
lap_times = []

def reset():
    """
    Reset the Player and Game state, pause 3 seconds
    """
    global player_velocity, player_angle, player_pos, seen_finish, prev_tile, amount_warnings, pause, tiles_visited, start_time, message
    player_velocity = 0
    player_angle = 90
    player_pos = pygame.Vector2(finish_tile[0] * TILE_SIZE, finish_tile[1] * TILE_SIZE + TILE_SIZE / 2)
    tiles_visited = []
    seen_finish = False
    start_time = None
    pause = 3
    prev_tile = None
    amount_warnings = 0
    message = ""

# Make the track
track_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
for x in range(tiles_x):
    for y in range(tiles_y):
        # Track is a list of rows. So first access y position then x position.
        tile = track[y][x]
        if tile == Tile.STRAIGHT_RIGHT:
            rotated_straight = pygame.transform.rotate(straight, 90)
            track_surface.blit(rotated_straight, (TILE_SIZE * x, TILE_SIZE * y))
        elif tile == Tile.STRAIGHT_LEFT:
            rotated_straight = pygame.transform.rotate(straight, -90)
            track_surface.blit(rotated_straight, (TILE_SIZE*x, TILE_SIZE*y))
        elif tile == Tile.STRAIGHT_UP:
            track_surface.blit(straight, (TILE_SIZE*x, TILE_SIZE*y))
        elif tile == Tile.STRAIGHT_DOWN:
            rotated_straight = pygame.transform.rotate(straight, 180)
            track_surface.blit(rotated_straight, (TILE_SIZE*x, TILE_SIZE*y))
        elif tile == Tile.CORNER_UP_RIGHT:
            track_surface.blit(corner, (TILE_SIZE*x, TILE_SIZE*y))
        elif tile == Tile.CORNER_UP_LEFT:
            rotated_corner = pygame.transform.rotate(corner, -90)
            track_surface.blit(rotated_corner, (TILE_SIZE*x, TILE_SIZE*y))
        elif tile == Tile.CORNER_DOWN_LEFT:
            rotated_corner = pygame.transform.rotate(corner, 180)
            track_surface.blit(rotated_corner, (TILE_SIZE*x, TILE_SIZE*y))
        elif tile == Tile.CORNER_DOWN_RIGHT:
            rotated_corner = pygame.transform.rotate(corner, -270)
            track_surface.blit(rotated_corner, (TILE_SIZE * x, TILE_SIZE*y))

# Load font
font = pygame.font.Font(None, 36)
start_time = None

# Game loop
while running:
    # Close program event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Set background color and draw track on screen
    screen.fill((116, 179, 74))
    screen.blit(track_surface, (0,0))

    # Draw the player
    rotated_surface = pygame.transform.rotate(player, player_angle)
    rotated_rect = rotated_surface.get_rect(center=player_pos)
    screen.blit(rotated_surface, rotated_rect)

    # Check for a pause
    if not pause:
        # Check input and act accordingly
        keys = pygame.key.get_pressed()
        rad = math.radians(player_angle) # Angle in radians
        player_velocity = min(0, player_velocity + friction * dt)
        dist = player_velocity * dt # The distance traveled
        player_pos.y += dist * math.cos(rad) # Change y position
        player_pos.x += dist * math.sin(rad) # Change x position
        if keys[pygame.K_UP]:
            player_velocity -= player_acceleration * dt # accelerate
            player_velocity = max(-player_max_velocity, player_velocity) # max velocity is
            if not start_time:
                start_time = pygame.time.get_ticks()
        if keys[pygame.K_DOWN]:
            player_velocity += player_deceleration * dt
            player_velocity = min(0, player_velocity)
        if keys[pygame.K_LEFT]:
            player_angle += 3
        if keys[pygame.K_RIGHT]:
            player_angle -= 3

        cur_tile = get_tile_pos(player_pos)
        cur_type = track[cur_tile[1]][cur_tile[0]]

        if cur_tile != prev_tile:
            prev_tile = cur_tile
            if cur_tile not in tiles_visited and cur_type != Tile.GRASS:
                tiles_visited.append(cur_tile)
                print(len(tiles_visited))
            if cur_tile == finish_tile:
                if not seen_finish:
                    seen_finish = True
                elif len(tiles_visited) < min_visited_tiles:
                    dsq = True
                else:

                    lap_times.append(pygame.time.get_ticks() - start_time + amount_warnings * 5000)
                    lap_times.sort()
                    lap_times = lap_times[:5]
                    reset()
            if cur_type == Tile.GRASS:
                amount_warnings += 1
                if amount_warnings == 3:
                    dsq = True

        if dsq:
            dsq = False
            message = "Lap INVALIDATED: "
            reset()
        else:
            if start_time:
                elapsed_ms = pygame.time.get_ticks() - start_time
                elapsed_sec = elapsed_ms // 1000
                elapsed_min = elapsed_sec // 60
                elapsed_sec = elapsed_sec % 60
                elapsed_ms = elapsed_ms % 1000
                timer_text = font.render(f"Time: {elapsed_min}:{elapsed_sec}:{elapsed_ms}", True, (255, 255, 255))
                screen.blit(timer_text, (screen_width/2 - timer_text.get_width()/2, 10))  # top-left corner
    else:
        pause = max(0, pause - dt)
        pause_text = font.render(f"{message} {math.ceil(pause)}", True, (255, 255, 255))
        screen.blit(pause_text, (screen_width/2 - pause_text.get_width()/2, screen_height/2))

    # Render lap times
    y = 10

    title = font.render("Lap times:", True, (255, 255, 255))
    screen.blit(title, (1200, y))
    y += 30  # move down

    for i in range(len(lap_times)):
        milliseconds = lap_times[i] % 1000
        seconds = (lap_times[i] // 1000) % 60
        minutes = lap_times[i] // 1000 // 60

        line = f"{i+1}. {minutes}:{seconds:02}:{milliseconds:03}"
        rendered = font.render(line, True, (255, 255, 255))
        screen.blit(rendered, (1200, y))

        y += 25  # spacing between lines

    # Render penalties
    penalty_text = font.render(f"({amount_warnings} warnings), Penalty: {amount_warnings * 5}s", True, (255, 255, 255))
    if amount_warnings > 0:
        screen.blit(penalty_text, (10, 10))

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()