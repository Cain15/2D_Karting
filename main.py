import pygame
import math
import random
from track_gen import read_track, Tile
import time

def get_tile_pos(pygame_pos):
    x = math.floor(pygame_pos.x / TILE_SIZE)
    y = math.floor(pygame_pos.y / TILE_SIZE)
    return x,y

screen_width = 1600
screen_height = 960

pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
running = True
dt = 0

TILE_SIZE = 80
tiles_x = 20
tiles_y = 12
finish_tile = (16,10)

player_pos = pygame.Vector2(finish_tile[0]*TILE_SIZE, finish_tile[1]*TILE_SIZE + TILE_SIZE/2)
player_angle = 90

player = pygame.image.load("assets/player.png")
player = pygame.transform.scale(player, (20,40))

player_acceleration = 35
player_deceleration = 100
player_velocity = 0

track = read_track('track1.tr')
corner = pygame.image.load('assets/corner.png')
corner = pygame.transform.scale(corner, (80,80))
straight = pygame.image.load('assets/straight.png')
straight = pygame.transform.scale(straight, (80,80))

seen_finish = False
prev_tile = None
min_visited_tiles = 69
tiles_visited = []
amount_warnings = 0
pause = 0
message = ""
dsq = False

lap_times = []

def reset():
    global player_velocity, player_angle, player_pos, seen_finish, prev_tile, amount_warnings, pause, tiles_visited, start_time
    player_velocity = 0
    player_angle = 90
    player_pos = pygame.Vector2(finish_tile[0] * TILE_SIZE, finish_tile[1] * TILE_SIZE + TILE_SIZE / 2)
    tiles_visited = []
    seen_finish = False
    start_time = None
    pause = 3
    prev_tile = None
    amount_warnings = 0

track_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

for x in range(tiles_x):
    for y in range(tiles_y):
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

font = pygame.font.Font(None, 36)
start_time = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((116, 179, 74))
    screen.blit(track_surface, (0,0))

    rotated_surface = pygame.transform.rotate(player, player_angle)
    rotated_rect = rotated_surface.get_rect(center=player_pos)

    screen.blit(rotated_surface, rotated_rect)

    if not pause:
        keys = pygame.key.get_pressed()
        rad = math.radians(player_angle)
        dist = player_velocity * dt
        player_pos.y += dist * math.cos(rad)
        player_pos.x += dist * math.sin(rad)
        if keys[pygame.K_UP]:
            player_velocity -= player_acceleration * dt
            player_velocity = max(-200, player_velocity)
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
            if cur_tile == finish_tile:
                if not seen_finish:
                    seen_finish = True
                elif len(tiles_visited) < min_visited_tiles:
                    dsq = True
                else:
                    lap_times.append(pygame.time.get_ticks() - start_time)
                    print(len(tiles_visited))
                    reset()

        if dsq:
            dsq = False
            message = "You were DISQUALIFIED: "
            print(tiles_visited)
            reset()


        if cur_type == Tile.GRASS:
            amount_warnings += 1
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

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()