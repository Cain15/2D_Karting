import pygame
import math

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_angle = 0

rect_surface = pygame.Surface((16, 32), pygame.SRCALPHA)
pygame.draw.rect(rect_surface, (255, 0, 0), (0, 0, 16, 32))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 200, 0))

    rotated_surface = pygame.transform.rotate(rect_surface, player_angle)
    rotated_rect = rotated_surface.get_rect(center=player_pos)

    screen.blit(rotated_surface, rotated_rect)


    keys = pygame.key.get_pressed()
    rad = math.radians(player_angle)
    if keys[pygame.K_UP]:
        player_pos.y -= 300 * math.cos(rad) * dt
        player_pos.x -= 300 * math.sin(rad) * dt
    if keys[pygame.K_DOWN]:
        player_pos.y += 300 * abs(math.cos(rad)) * dt
        player_pos.x += 300 * abs(math.sin(rad)) * dt
    if keys[pygame.K_LEFT]:
        player_angle += 4
    if keys[pygame.K_RIGHT]:
        player_angle -= 4

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()