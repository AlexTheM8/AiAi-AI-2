import datetime
import logging
import math
from ctypes import windll
from enum import Enum
from optparse import OptionParser
from os import listdir, mkdir, path
from pickle import dump
from time import perf_counter, sleep
import cv2
import dolphin_memory_engine as mem_eng
import neat
import numpy as np
import pygetwindow as gw
from mss import mss

from controller import Controller

# MEMORY INFO REF SHEET
# https://github.com/CraftedCart/smbinfo/tree/master/SMB2


class LogOptions(Enum):
    FULL = 'full'
    PARTIAL = 'partial'
    NONE = 'none'


def create_logger(option):
    log = logging.getLogger("Aiai_AI")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False

    if option != LogOptions.NONE.value:
        console_handle = logging.StreamHandler()
        console_handle.setLevel(logging.DEBUG)

        log_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
        console_handle.setFormatter(log_format)

        log.addHandler(console_handle)
    return log


def get_img():
    with mss() as sct:
        return cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_RGB2BGR)


def get_pos():
    return mem_eng.read_float(0x805BC9B0), mem_eng.read_float(0x805BC9B4), mem_eng.read_float(0x805BC9B8)


def get_goal_pos():
    return mem_eng.read_float(0x805E8DE8), mem_eng.read_float(0x805D5040), mem_eng.read_float(0x805E8DF0)


def get_state():
    return mem_eng.read_bytes(0x805E914C, 9).strip(b'\x00').decode("utf-8")


def calc_distance(pos1, pos2):
    p1 = np.array(pos1)
    p2 = np.array(pos2)

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def calc_angle(pPos, gPos):
    view_angle = 360 * mem_eng.read_byte(0x8054E072) / 254
    px, py, pz = pPos
    gx, gy, gz = gPos
    zDiff = gz - pz
    xDiff = gx - px
    if xDiff == 0:
        angle = 0 if zDiff < 0 else 180
    else:
        angle = math.atan(zDiff / xDiff)
    adj = view_angle - angle
    return adj if adj < 180 else adj - 360


def interpret_and_act(curr_pos, x_input, y_input, st, g_max):
    done, info = False, ''

    controller.do_movement(x_input, y_input)

    state = get_state()
    match state:
        case 'FALL OUT' | 'TIME OVER':
            g_max -= 25
            done, info = True, state
        case 'GOAL':
            g_max = 30 + (1.25 * (60 - (perf_counter() - st)))  # [30, 105]
            done, info = True, state
        case _:
            g_max = max(g_max, 50 - calc_distance(goal_pos, curr_pos))
    return g_max, done, info


def conduct_genome(genome, cfg, genome_id, pop=None):
    global p

    p = p if pop is None else pop

    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, cfg)

    current_max_fitness, g_max, zero_step, done, info = float('-inf'), float('-inf'), 0, False, ''

    # Clear out old state
    mem_eng.write_bytes(0x805E914C, b'\x00')
    mem_eng.write_float(0x805BC9B0, start_pos[0])
    mem_eng.write_float(0x805BC9B4, start_pos[1])
    mem_eng.write_float(0x805BC9B8, start_pos[2])
    sleep(0.1)
    controller.load_state()
    sleep(0.1)
    controller.load_state()

    if options.logging == LogOptions.FULL.value:
        logger.info(f'running genome {genome_id} in generation {p.generation}')
    st = perf_counter()
    while not done:
        # get next image & pos
        curr_pos = get_pos()
        px, py, pz = curr_pos
        gx, gy, gz = goal_pos

        # Distance to goal, horizontal angle to goal, yDiff to goal
        additional_inputs = [calc_distance(curr_pos, goal_pos), calc_angle(curr_pos, goal_pos), py - gy]
        img = get_img()

        img_copy = cv2.resize(img, (inx, iny))
        img_copy = np.reshape(img_copy, (inx, iny, inc))

        img_array = np.ndarray.flatten(img_copy)
        full_array = np.append(img_array, additional_inputs)

        # Get end result input to game
        x_input, y_input = net.activate(full_array)

        g_max, done, info = interpret_and_act(curr_pos, x_input, y_input, st, g_max)

        if info != '' and options.logging == LogOptions.FULL.value:
            logger.info(f'{info}')

        # Guarantee significant change
        if g_max > current_max_fitness and abs(g_max - current_max_fitness) > 0.0001:
            current_max_fitness = g_max
            zero_step = 0
        elif options.zero_kill:
            if calc_distance(curr_pos, get_pos()) < 0.001:
                zero_step += 60
            elif calc_distance(start_pos, get_pos()) < 15:
                zero_step += 5
            else:
                zero_step = 0

        if not done and zero_step > max_steps:
            done = True
            if options.logging == LogOptions.FULL.value:
                logger.info('Timed out due to stagnation')
            g_max -= 25
        genome.fitness = g_max
    logger.info(f'generation: {p.generation}, genome: {genome_id}, fitness: {genome.fitness}')
    if info == 'GOAL':
        update_records(p.generation, 'GOAL', genome.fitness, stats_path)
    controller.do_movement(0, 0)  # Reset movement
    return genome.fitness


def update_stats(gen, sr, stat_dir, file='stats.csv'):
    if not path.isdir(stat_dir):
        mkdir(stat_dir)
    with open(f'{stat_dir}/{file}', 'a') as f:
        f.write(','.join([str(gen), str(max_fitness[gen]), str(sr.get_fitness_mean()[-1]),
                          str(sr.get_fitness_stdev()[-1])]) + '\n')


def update_records(gen, rec_type, fitness, stat_dir, file='records.csv'):
    if not path.isdir(stat_dir):
        mkdir(stat_dir)
    with open(f'{stat_dir}/{file}', 'a') as f:
        timestamp = datetime.datetime.now()
        f.write(','.join([str(timestamp), rec_type, str(gen), str(fitness)]) + '\n')


def get_curr_max_fitness(stat_dir, file='max_fitness.txt'):
    if not path.isdir(stat_dir):
        mkdir(stat_dir)
    try:
        with open(f'{stat_dir}/{file}', 'r') as f:
            return float(f.read())
    except:
        return float('-inf')


def set_curr_max_fitness(stat_dir, fitness, file='max_fitness.txt'):
    if not path.isdir(stat_dir):
        mkdir(stat_dir)
    with open(f'{stat_dir}/{file}', 'w+') as f:
        f.write(str(fitness))


def eval_genomes(genomes, cfg):
    global curr_max_fitness
    if len(stat_reporter.get_fitness_mean()) > 0 and options.stats:
        update_stats(p.generation - 1, stat_reporter, stats_path)
    max_fit = float('-inf')
    for genome_id, genome in genomes:
        fit = conduct_genome(genome, cfg, genome_id)
        max_fit = max(max_fit, fit)
        if max_fit > curr_max_fitness:
            update_records(p.generation, 'RECORD', max_fit, stats_path)
            set_curr_max_fitness(stats_path, max_fit)
            curr_max_fitness = max_fit
    max_fitness[p.generation] = max_fit


# TODO Figure out playback settings
if __name__ == '__main__':
    parser = OptionParser()

    parser.set_defaults(stats=True, zero_kill=True)

    parser.add_option('-l', '--logging', dest='logging', choices=[o.value for o in LogOptions],
                      help='Logging options: [full, partial, none]. (Default=full)', default=LogOptions.FULL.value)
    parser.add_option('-s', '--stats', dest='stats',
                      help='Enable this flag to stop saving evolution stats. (Default=true)', action='store_false')
    parser.add_option('-z', '--zero_kill', dest='zero_kill',
                      help='Enable this flag to stop killing genome at 0mph. (Default=true)', action='store_false')

    options, args = parser.parse_args()

    # Controller
    controller = Controller()

    # Image setup
    user32 = windll.user32
    user32.SetProcessDPIAware()
    windows = gw.getWindowsWithTitle('Super Monkey Ball 2')
    if len(windows) == 0:
        print('ERROR: Could not find SMB window')
        exit(-1)
    window = windows[0]
    pad, top_pad = 10, 30

    bbox = window.box
    monitor = {"top": bbox.top + pad + top_pad, "left": bbox.left + pad,
               "width": bbox.width - (pad * 2), "height": bbox.height - (pad * 2) - top_pad}
    scale = 14
    inx, iny, inc = bbox.width // scale, bbox.height // scale, 3

    max_steps = 5000
    max_fitness = {}

    logger = create_logger(options.logging)

    mem_eng.hook()
    if not mem_eng.is_hooked():
        print('ERROR: Could not hook into memory')
        exit(-1)
    sleep(1)
    controller.load_state()
    sleep(1)
    stage = mem_eng.read_bytes(0x805BDA10, 15).strip(b'\x00').decode("utf-8").lower()
    stats_path = f'stats/{stage}'
    curr_max_fitness = get_curr_max_fitness(stats_path)

    # Network setup
    hist_path = f'history/{stage}'
    checkpointer = neat.Checkpointer(generation_interval=1, filename_prefix=f'{hist_path}/neat-checkpoint-')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')
    if not path.isdir(hist_path):
        mkdir(hist_path)
    if len(listdir(hist_path)) > 0:
        m = max([int(f[f.rfind('-') + 1:]) for f in listdir(hist_path)])
        p = checkpointer.restore_checkpoint(f'{hist_path}/neat-checkpoint-{m}')
        p.generation += 1
        logger.info(f'Restoring checkpoint {m} for stage {stage}')
        p.config = config
    else:
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stat_reporter = neat.StatisticsReporter()
    p.add_reporter(stat_reporter)
    p.add_reporter(checkpointer)

    goal_pos = get_goal_pos()
    start_pos = get_pos()
    # Final
    winner = p.run(eval_genomes)

    win_path = 'winners'
    if not path.isdir(win_path):
        mkdir(win_path)

    with open(f'{win_path}/{stage}_winner.pkl', 'wb') as output:
        dump(winner, output, 1)
