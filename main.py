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

def out_of_bounds(pygame_pos):
    """
    Check if the position is out of bounds
    :param pygame_pos: The positon to be checked
    :return: true if out of bounds, false if not
    """
    if pygame_pos.x < 0 or pygame_pos.x > screen_width:
        return True
    if pygame_pos.y < 0 or pygame_pos.y > screen_height:
        return True
    return False

def write_laptime(laptime, dsq = False):
    seconds = laptime / 1000
    text = str(seconds)
    if dsq:
        text += ",invalid\n"
    else:
        text += ",valid\n"
    with open("laptime.txt", "a") as f:
        f.write(text)


def is_boundary(tile_type: Tile, movement_dir: str) -> bool:
    """
    Return True if the tile is considered a boundary for the ray moving in movement_dir.
    """
    # Straights
    if tile_type == Tile.STRAIGHT_UP and movement_dir in ["LEFT", "RIGHT", "UP"]:
        return True
    if tile_type == Tile.STRAIGHT_DOWN and movement_dir in ["LEFT", "RIGHT", "DOWN"]:
        return True
    if (tile_type == Tile.STRAIGHT_LEFT or tile_type == Tile.FINISH_LINE) and movement_dir in ["UP", "DOWN", "RIGHT"]:
        return True
    if tile_type == Tile.STRAIGHT_RIGHT and movement_dir in ["UP", "DOWN", "LEFT"]:
        return True

    if tile_type == Tile.CORNER_DOWN_RIGHT and movement_dir in ["DOWN", "LEFT"]:
        return True
    if tile_type == Tile.CORNER_DOWN_LEFT and movement_dir in ["DOWN", "RIGHT"]:
        return True
    if tile_type == Tile.CORNER_UP_RIGHT and movement_dir in ["UP", "LEFT"]:
        return True
    if tile_type == Tile.CORNER_UP_LEFT and movement_dir in ["UP", "RIGHT"]:
        return True


    return False  # Otherwise, tile is "passable" in ray direction

def ray_trace_bound(angle):
    ray = player_pos.copy()
    ray_rad = math.radians(-player_angle + angle)
    direction = pygame.Vector2(math.sin(ray_rad), -math.cos(ray_rad))
    direction = direction.normalize()
    prev_ray_tile = get_tile_pos(ray)
    ray_tile_type = track[prev_ray_tile[1]][prev_ray_tile[0]]
    step = 1.0
    boundary_hit = False
    movement_dir = None
    while not boundary_hit:
        ray += direction * step
        if out_of_bounds(ray):
            boundary_hit = True
        else:
            ray_tile = get_tile_pos(ray)
            if ray_tile != prev_ray_tile:
                movement_dir = None
                if ray_tile[0] > prev_ray_tile[0]:
                    movement_dir = "RIGHT"
                elif ray_tile[0] < prev_ray_tile[0]:
                    movement_dir = "LEFT"
                elif ray_tile[1] > prev_ray_tile[1]:
                    movement_dir = "DOWN"
                elif ray_tile[1] < prev_ray_tile[1]:
                    movement_dir = "UP"

                if is_boundary(ray_tile_type, movement_dir):
                    boundary_hit = True
                else:
                    prev_ray_tile = ray_tile
                    ray_tile_type = track[ray_tile[1]][ray_tile[0]]
                    if ray_tile_type == Tile.GRASS:
                        boundary_hit = True
    dx, dy = direction.x, direction.y
    correction_pixels = 7
    t = 0
    if movement_dir in ["LEFT", "RIGHT"]:
        # Need vertical (Y) correction
        t = correction_pixels / abs(dx)

    elif movement_dir in ["UP", "DOWN"]:
        # Need horizontal (X) correction
        t = correction_pixels / abs(dy)
    ray -= direction * t
    return ray

def get_reward(current_tile, previous_tile):
    if current_tile == previous_tile:
        return 0
    if track[current_tile[1]][current_tile[0]] == Tile.GRASS:
        return -0.5
    if track[current_tile[1]][current_tile[0]] == Tile.FINISH_LINE and seen_finish:
        if len(tiles_visited) < min_visited_tiles:
            return -1
        else:
            return 10
    if current_tile not in tiles_visited:
        return 1
    return -1



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
offset_angle = 180
player_pos = pygame.Vector2((finish_tile[0]+1)*TILE_SIZE, finish_tile[1]*TILE_SIZE + TILE_SIZE/2)
player_angle = 90
player = pygame.image.load("assets/player.png")
player = pygame.transform.scale(player, (20,40))
player = pygame.transform.rotate(player, offset_angle)
friction = 0.1
player_acceleration = 55
player_deceleration = 108
player_velocity = 0
player_max_velocity = 270

# Track generation
track = read_track('track1.tr')
corner = pygame.image.load('assets/corner.png')
corner = pygame.transform.scale(corner, (80,80))
straight = pygame.image.load('assets/straight.png')
straight = pygame.transform.scale(straight, (80,80))
finish = pygame.image.load('assets/finish.png')
finish = pygame.transform.scale(finish, (80,80))

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
    global player_velocity, player_angle, player_pos, seen_finish, prev_tile, amount_warnings, pause, tiles_visited, start_time
    player_velocity = 0
    player_angle = 90
    player_pos = pygame.Vector2((finish_tile[0]+1) * TILE_SIZE, finish_tile[1] * TILE_SIZE + TILE_SIZE / 2)
    tiles_visited = []
    seen_finish = False
    start_time = None
    pause = 3
    prev_tile = None
    amount_warnings = 0

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
        elif tile == Tile.FINISH_LINE:
            rotated_finish = pygame.transform.rotate(finish, -90)
            track_surface.blit(rotated_finish, (TILE_SIZE * x, TILE_SIZE * y))

# Load font
font = pygame.font.Font(None, 36)
start_time = None


from AIModel import QAgent, Action
AI_mode = True
model = QAgent()
prev_state = None
prev_action = None

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
        rad = math.radians(player_angle) # Angle in radians
        player_velocity *= (1 - friction * dt)
        if player_velocity > -0.5:
            player_velocity = 0
        dist = player_velocity * dt # The distance traveled
        player_pos.y += dist * math.cos(rad) # Change y position
        player_pos.x += dist * math.sin(rad) # Change x position
        if abs(player_velocity) > 5:
            player_velocity *= 0.999  # tiny stabilizer

        turn_speed = 180  # degrees per second at low speed
        speed_factor = 1 / (1 + abs(player_velocity) * 0.01)

        if out_of_bounds(player_pos):
            dsq = True
            message = "Your car exploded! You died! :( "
        else:
            cur_tile = get_tile_pos(player_pos)
            cur_type = track[cur_tile[1]][cur_tile[0]]
            if AI_mode:
                # Get the inputs for AI
                # Inputs: player_velocity, player_angle, 10 Track edge points
                # Calculate the track edge points
                v_norm = player_velocity / player_max_velocity
                theta_norm = player_angle / 360
                features = [v_norm, theta_norm]
                screen_diag = math.hypot(screen_width, screen_height)
                for i in range(11):
                    cur_ray = ray_trace_bound(-100 + i * 20)
                    # pygame.draw.line(screen, (255, 255, 255), player_pos, cur_ray, 2)
                    features.append(player_pos.distance_to(cur_ray)/screen_diag)

                if prev_state:
                    reward = get_reward(cur_tile, prev_tile)
                    model.update(prev_state, prev_action, reward, features)

                act = model.act(features)
                action = Action(act[0]), Action(act[1])

                prev_state = features
                prev_action = act
                if action[1] == Action.accelerate:
                    player_velocity -= player_acceleration * dt  # accelerate
                    player_velocity = max(-player_max_velocity, player_velocity)  # max velocity is
                    if not start_time:
                        start_time = pygame.time.get_ticks()
                elif action[1] == Action.decelerate:
                    player_velocity += player_deceleration * dt
                    player_velocity = min(0, player_velocity)
                if action[0] == Action.left:
                    player_angle += turn_speed * speed_factor * dt
                elif action[0] == Action.right:
                    player_angle -= turn_speed * speed_factor * dt
            else:
                # Check input and act accordingly
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    player_velocity -= player_acceleration * dt # accelerate
                    player_velocity = max(-player_max_velocity, player_velocity) # max velocity is
                    if not start_time:
                        start_time = pygame.time.get_ticks()
                if keys[pygame.K_DOWN]:
                    player_velocity += player_deceleration * dt
                    player_velocity = min(0, player_velocity)

                if keys[pygame.K_LEFT]:
                    player_angle += turn_speed * speed_factor * dt
                if keys[pygame.K_RIGHT]:
                    player_angle -= turn_speed * speed_factor * dt

            player_angle %= 360
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

                        lap_times.append(pygame.time.get_ticks() - start_time + amount_warnings * 5000)
                        lap_times.sort()
                        lap_times = lap_times[:5]
                        # write_laptime(pygame.time.get_ticks() - start_time + amount_warnings * 5000)
                        reset()
                if cur_type == Tile.GRASS:
                    amount_warnings += 1
                    if amount_warnings == 3:
                        dsq = True

        if dsq:
            dsq = False
            if not message:
                message = "Lap INVALIDATED: "
            # write_laptime(pygame.time.get_ticks() - start_time, True)
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
        screen.blit(pause_text, (screen_width/2 - pause_text.get_width()/2 + 100, screen_height/2 - 100))
        if pause == 0:
            message = ""

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
    penalty_text = font.render(f"[{amount_warnings} warning(s)], Penalty: {amount_warnings * 5}s", True, (255, 255, 255))
    if amount_warnings > 0:
        screen.blit(penalty_text, (10, 10))

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()