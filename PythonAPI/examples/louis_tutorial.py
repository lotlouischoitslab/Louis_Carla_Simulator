import glob
import os
import sys
import argparse

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time

actor_list = []

try:
    louis_computer_host = '127.0.0.1'
    client = carla.Client(louis_computer_host,2000)
    client.set_timeout(2.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0] #filter the model by model3
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points()) #random spawn points

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True) 



finally:
    for actor in actor_list:
        actor.destroy()
    print('All cleaned up!')


