import pygame
import math
import random

pygame.init()
screen = pygame.display.set_mode((1280, 736))
clock = pygame.time.Clock()
running = True
dt = 0
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_angle = 0

rect_surface = pygame.Surface((16, 32), pygame.SRCALPHA)
pygame.draw.rect(rect_surface, (255, 0, 0), (0, 0, 16, 32))

player_acceleration = 100
player_deceleration = 300
player_velocity = 0

grass_image = pygame.image.load('assets/grass.png').convert()
TILE_SIZE = 32
tiles_x = int(1280 / TILE_SIZE)
tiles_y = int(736 / TILE_SIZE)

# Precompute a random rotation for each tile
tile_rotations = [
    [random.choice([0, 90, 180, 270]) for y in range(tiles_y)]
    for x in range(tiles_x)
]


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 200, 0))
    for x in range(tiles_x):
        for y in range(tiles_y):
            angle = tile_rotations[x][y]
            rotated_grass = pygame.transform.rotate(grass_image, angle)

            # Adjust position because rotation can change surface size
            rect = rotated_grass.get_rect(topleft=(32 * x, 32 * y))
            screen.blit(rotated_grass, rect)

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