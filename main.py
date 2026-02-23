import pygame
import math
import random
from track_gen import read_track, Tile

pygame.init()
screen = pygame.display.set_mode((1280, 768))
clock = pygame.time.Clock()
running = True
dt = 0
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_angle = 0

rect_surface = pygame.Surface((8, 16), pygame.SRCALPHA)
pygame.draw.rect(rect_surface, (255, 0, 0), (0, 0, 8, 16))

player_acceleration = 35
player_deceleration = 100
player_velocity = 0

track = read_track('track1.tr')
corner = pygame.image.load('assets/corner.png')
straight = pygame.image.load('assets/straight.png')

TILE_SIZE = 64
tiles_x = int(1280 / TILE_SIZE)
tiles_y = int(768 / TILE_SIZE)

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



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((116, 179, 74))
    screen.blit(track_surface, (0,0))

    rotated_surface = pygame.transform.rotate(rect_surface, player_angle)
    rotated_rect = rotated_surface.get_rect(center=player_pos)

    screen.blit(rotated_surface, rotated_rect)


    keys = pygame.key.get_pressed()
    rad = math.radians(player_angle)
    dist = player_velocity * dt
    player_pos.y += dist * math.cos(rad)
    player_pos.x += dist * math.sin(rad)
    if keys[pygame.K_UP]:
        player_velocity -= player_acceleration * dt
        player_velocity = max(-300, player_velocity)
    if keys[pygame.K_DOWN]:
        player_velocity += player_deceleration * dt
        player_velocity = min(0, player_velocity)
    if keys[pygame.K_LEFT]:
        player_angle += 4
    if keys[pygame.K_RIGHT]:
        player_angle -= 4

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()