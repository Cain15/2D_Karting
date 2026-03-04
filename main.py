import numpy as np
import pygame
import math
from track_gen import read_track, Tile, track_walk, generate_corner_waypoints, corner_reward
import threading

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

def ray_trace_bound(player, angle):
    MAX_RAY_DIST = 300  # pixels
    ray = player.player_pos.copy()
    ray_rad = math.radians(-player.player_angle + angle)
    direction = pygame.Vector2(math.sin(ray_rad), -math.cos(ray_rad))
    prev_ray_tile = get_tile_pos(ray)
    ray_tile_type = track[prev_ray_tile[1]][prev_ray_tile[0]]
    step = 1.0
    steps = 0
    boundary_hit = False
    movement_dir = None
    while not boundary_hit and steps < MAX_RAY_DIST:
        ray += direction * step
        steps += 1
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
        # Need vertical (Y) correctiondt
        t = correction_pixels / abs(dx)

    elif movement_dir in ["UP", "DOWN"]:
        # Need horizontal (X) correction
        t = correction_pixels / abs(dy)
    ray -= direction * t
    return ray

def get_reward(current_tile, previous_tile, player):
    if track[current_tile[1]][current_tile[0]] == Tile.GRASS:
        return -50.0
    if previous_tile is None:
        return 0.0
    try:
        cur_idx = tile_index[current_tile]
        prev_idx = tile_index[previous_tile]
    except ValueError:
        # In case something weird happens
        print("Something went wrong", current_tile, previous_tile)
        return -50.0
    track_len = len(tile_order)
    delta = cur_idx - prev_idx
    track_progress = tile_index.get(current_tile, 0) / len(tile_order)
    progress_multiplier = 1.0 + 1.5 * max(0.0, track_progress - 0.7)

    # Wrap-around correction
    if delta < -track_len / 2:
        delta += track_len
    elif delta > track_len / 2:
        delta -= track_len
    if delta > 1:
        rew = -50
    else:
        rew = float(delta) * 5 * progress_multiplier
    if abs(p.player_velocity) < 5:
        rew -= 0.05

    speed_bonus = abs(p.player_velocity) / p.player_max_velocity
    # rew += 0.01 * speed_bonus
    rew += corner_reward(player, waypoints)

    return rew



# Screen size
screen_width = 1600
screen_height = 960

# Initialize window
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
running = True
dt = 0

# Tile variables
TILE_SIZE = 80
tiles_x = 20
tiles_y = 12
finish_tile = (4,10)

# Player properties
from Player import Player
offset_angle = 180
player_start_pos = pygame.Vector2((finish_tile[0]+1)*TILE_SIZE, finish_tile[1]*TILE_SIZE + TILE_SIZE/2)
player = pygame.image.load("assets/player.png").convert_alpha()
player = pygame.transform.scale(player, (20,40))
player = pygame.transform.rotate(player, offset_angle)
friction = 0.1

# Track generation
track = read_track('track1.tr')
corner = pygame.image.load('assets/corner.png').convert_alpha()
corner = pygame.transform.scale(corner, (80,80))
straight = pygame.image.load('assets/straight.png').convert_alpha()
straight = pygame.transform.scale(straight, (80,80))
finish = pygame.image.load('assets/finish.png').convert_alpha()
finish = pygame.transform.scale(finish, (80,80))

# Progress tracking
min_visited_tiles = 68
tile_order = track_walk(track, (5,10))
tile_index = {tile: i for i, tile in enumerate(tile_order)}
waypoints = generate_corner_waypoints(track, tile_order)

# Game state variables
pause = 0
message = ""

# Keep lap times
lap_times = [42000]

def reset(play):
    """
    Reset the Player and Game state, pause 3 seconds
    """
    global pause
    play.player_velocity = 0
    play.player_angle = 90
    play.player_pos = pygame.Vector2((finish_tile[0] + 1) * TILE_SIZE, finish_tile[1] * TILE_SIZE + TILE_SIZE / 2)
    play.tiles_visited = []
    play.seen_finish = False
    play.start_time = None
    pause = 3 if not AI_mode else 0
    play.prev_tile = None
    play.amount_warnings = 0
    play.prev_action = None
    play.prev_state = None
    play.current_waypoint_index = 0
    play.prev_corner_distance = float("inf")

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


from AIModel import DDQNAgent, Action
AI_mode = True
model = DDQNAgent()
model.load()
if AI_mode:
    players = [Player(player_start_pos, 0.05)]
else:
    players = [Player(player_start_pos)]

# fps_timer = 0

rotated_cache = {}
clock = pygame.time.Clock()

# Game loop
while running:
    dt = min(clock.tick(60) / 1000, 0.05)
    # Close program event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Set background color and draw track on screen
    screen.fill((116, 179, 74))
    screen.blit(track_surface, (0,0))

    farthest_player = None
    farthest_tile_idx = 0
    for p in players:
        tile = get_tile_pos(p.player_pos)
        if not track[tile[1]][tile[0]] == Tile.GRASS:
            idx = tile_index[tile]
            if not farthest_player:
                farthest_player = p
                farthest_tile_idx = idx
            elif idx > farthest_tile_idx:
                farthest_player = p
                farthest_tile_idx = idx
        else:
            farthest_player = players[0]

    # waypoint = waypoints[farthest_player.current_waypoint_index]
    # pygame.draw.circle(screen, (255, 255, 255), (waypoint.x, waypoint.y), 10)

    # Draw the player
    for p in players:
        angle_key = int(p.player_angle) % 360
        if angle_key not in rotated_cache:
            rotated_cache[angle_key] = pygame.transform.rotate(player, angle_key)
        rotated_surface = rotated_cache[angle_key]
        rotated_rect = rotated_surface.get_rect(center=p.player_pos)
        screen.blit(rotated_surface, rotated_rect)

    # Check for a pause
    if not pause:
        for p in players:
            rad = math.radians(p.player_angle) # Angle in radians # Check for a pause
            p.player_velocity *= (1 - friction * dt)
            if p.player_velocity > -0.5:
                p.player_velocity = 0
            dist = p.player_velocity * dt # The distance traveled
            p.player_pos.y += dist * math.cos(rad) # Change y position
            p.player_pos.x += dist * math.sin(rad) # Change x position
            if abs(p.player_velocity) > 5:
                p.player_velocity *= 0.999  # tiny stabilizer

            turn_speed = 180  # degrees per second at low speed
            speed_factor = 1 / (1 + abs(p.player_velocity) * 0.01)

            if out_of_bounds(p.player_pos):
                p.dsq = True
                message = "Your car exploded! You died! :( "
            else:
                cur_tile = get_tile_pos(p.player_pos)
                cur_type = track[cur_tile[1]][cur_tile[0]]
                if AI_mode:
                    # Get the inputs for AI
                    # Inputs: player_velocity, player_angle, 10 Track edge points
                    # Calculate the track edge points
                    v_norm = p.player_velocity / p.player_max_velocity
                    theta = math.radians(p.player_angle)
                    wp = waypoints[p.current_waypoint_index]
                    to_wp = pygame.Vector2(wp.x, wp.y) - p.player_pos
                    forward = pygame.Vector2(math.sin(theta), -math.cos(theta))
                    if to_wp.length() > 0:
                        to_wp_norm = to_wp.normalize()
                        angle_diff = forward.angle_to(to_wp_norm) / 180.0
                    else:
                        angle_diff = 0.0
                    try:
                        tile_idx = tile_index[cur_tile]
                        progress = tile_idx/len(tile_order)
                    except KeyError:
                        progress = 0
                    features = [v_norm, math.sin(theta), math.cos(theta), angle_diff, progress]
                    for angle in [-90, -70, -50, -35, -20, -10, 0, 10, 20, 35, 50, 70, 90]:
                        cur_ray = ray_trace_bound(p, angle)
                        # pygame.draw.line(screen, (255, 255, 255), p.player_pos, cur_ray, 2)
                        features.append(np.clip(p.player_pos.distance_to(cur_ray)/300,0.0, 1.0))


                    act = model.act(features, p.eps)
                    action = Action(act[0]), Action(act[1])

                    if p.prev_state:
                        reward = get_reward(cur_tile, p.prev_tile, p)
                        lap_done = cur_tile == finish_tile and p.seen_finish and len(p.tiles_visited) >= min_visited_tiles
                        done = cur_type == Tile.GRASS or lap_done
                        if not done:
                            if p.prev_tile:
                                cur_id = tile_index[cur_tile]
                                prev_id = tile_index[p.prev_tile]
                                delta = cur_id - prev_id
                                track_len = len(tile_order)
                                if delta < -track_len / 2:
                                    delta += track_len
                                elif delta > track_len / 2:
                                    delta -= track_len
                                done = delta > 1
                        model.update(p.prev_state, p.prev_action, reward, features, done)
                        p.reward += reward
                        if done:
                            # print(p.reward)
                            p.reward = 0

                    p.prev_state = features
                    p.prev_action = act
                    if action[1] == Action.accelerate:
                        p.player_velocity -= p.player_acceleration * dt  # accelerate
                        p.player_velocity = max(-p.player_max_velocity, p.player_velocity)  # max velocity
                        if not p.start_time:
                            p.start_time = pygame.time.get_ticks()
                    elif action[1] == Action.decelerate:
                        p.player_velocity += p.player_deceleration * dt
                        p.player_velocity = min(0, p.player_velocity)
                    if action[0] == Action.left:
                        if abs(p.player_velocity) > 0:
                            p.player_angle += turn_speed * speed_factor * dt
                    elif action[0] == Action.right:
                        if abs(p.player_velocity) > 0:
                            p.player_angle -= turn_speed * speed_factor * dt
                else:
                    # Check input and act accordingly
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]:
                        p.player_velocity -= p.player_acceleration * dt # accelerate
                        p.player_velocity = max(-p.player_max_velocity, p.player_velocity) # max velocity is
                        if not p.start_time:
                            p.start_time = pygame.time.get_ticks()
                    if keys[pygame.K_DOWN]:
                        p.player_velocity += p.player_deceleration * dt
                        p.player_velocity = min(0, p.player_velocity)

                    if keys[pygame.K_LEFT]:
                        if abs(p.player_velocity) > 0:
                            p.player_angle += turn_speed * speed_factor * dt
                    if keys[pygame.K_RIGHT]:
                        if abs(p.player_velocity) > 0:
                            p.player_angle -= turn_speed * speed_factor * dt

                p.player_angle %= 360
                if cur_tile != p.prev_tile:
                    if p.prev_tile and not cur_type == Tile.GRASS and track[p.prev_tile[1]][p.prev_tile[0]] != Tile.GRASS:
                        cur_id = tile_index[cur_tile]
                        prev_id = tile_index[p.prev_tile]
                        delta = cur_id - prev_id
                        track_len = len(tile_order)
                        if delta < -track_len / 2:
                            delta += track_len
                        elif delta > track_len / 2:
                            delta -= track_len
                        if delta > 1:
                            p.dsq = True
                    p.prev_tile = cur_tile
                    if cur_tile not in p.tiles_visited and cur_type != Tile.GRASS:
                        p.tiles_visited.append(cur_tile)
                    if cur_tile == finish_tile:
                        if not p.seen_finish:
                            p.seen_finish = True
                        elif len(p.tiles_visited) < min_visited_tiles:
                            p.dsq = True
                        else:
                            new_time = pygame.time.get_ticks() - p.start_time + p.amount_warnings * 5000
                            if len(lap_times) == 0 or new_time < lap_times[0]:
                                model.save("ddqn_best.pth")
                            lap_times.append(new_time)
                            lap_times.sort()
                            lap_times = lap_times[:5]
                            # write_laptime(pygame.time.get_ticks() - start_time + amount_warnings * 5000)
                            reset(p)
                    elif cur_type == Tile.GRASS:
                        p.amount_warnings += 1
                        if p.amount_warnings == 3 or (AI_mode and p.amount_warnings == 1):
                            p.dsq = True

        # if start_time:
        #     if (pygame.time.get_ticks() - start_time) // 1000 > 300:
        #         dsq = True
            if p.dsq:
                p.dsq = False
                if not message:
                    message = "Lap INVALIDATED: "
                # write_laptime(pygame.time.get_ticks() - start_time, True)
                reset(p)
        else:
            if farthest_player.start_time:
                elapsed_ms = pygame.time.get_ticks() - farthest_player.start_time
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
    penalty_text = font.render(f"[{players[0].amount_warnings} warning(s)], Penalty: {players[0].amount_warnings * 5}s", True, (255, 255, 255))
    if players[0].amount_warnings > 0:
        screen.blit(penalty_text, (10, 10))

    pygame.display.flip()
    # fps_timer += dt
    # if fps_timer >= 1.0:
    #     print(clock.get_fps())
    #     fps_timer = 0

pygame.quit()