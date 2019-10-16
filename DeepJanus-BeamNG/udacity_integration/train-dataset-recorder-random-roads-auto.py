import json
import random
from typing import Tuple, List

import numpy as np

from core.folder_storage import SeedStorage
from self_driving import road_storage, beamng_member
from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.decal_road import DecalRoad
from self_driving.road_generator import RoadGenerator
from self_driving.road_points import RoadPoints
from self_driving.road_polygon import RoadPolygon
from udacity_integration.training_data_collector_and_writer import TrainingDataCollectorAndWriter
from self_driving.utils import get_node_coords, points_distance

from beamngpy.vehicle import Vehicle
from beamngpy.scenario import Scenario
from beamngpy import BeamNGpy

random.seed(42)
np.random.seed(42)


def get_node_coords(node):
    return node[0], node[1], node[2]


def get_rotation(road: DecalRoad):
    v1 = road.nodes[0][:2]
    v2 = road.nodes[1][:2]
    v = np.subtract(v1, v2)
    deg = np.degrees(np.arctan2([v[0]], [v[1]]))
    return (0, 0, deg)


def distance(p1, p2):
    return np.linalg.norm(np.subtract(get_node_coords(p1), get_node_coords(p2)))


def run_sim(street_1: DecalRoad):
    waypoints = []
    for node in street_1.nodes:
        waypoint = BeamNGWaypoint("waypoint_" + str(node), get_node_coords(node))
        waypoints.append(waypoint)

    print(len(waypoints))
    maps.beamng_map.generated().write_items(
        street_1.to_json() + '\n' + "\n".join([waypoint.to_json() for waypoint in waypoints]))

    beamng = BeamNGpy('localhost', 64256)
    scenario = Scenario('tig', 'tigscenario')

    vehicle = Vehicle('ego_vehicle', model='etk800', licence='TIG', color='Red')

    sim_data_collector = TrainingDataCollectorAndWriter(vehicle, beamng, street_1)

    scenario.add_vehicle(vehicle, pos=get_node_coords(street_1.nodes[0]), rot=get_rotation(street_1))

    scenario.make(beamng)

    beamng.open()
    beamng.set_deterministic()
    beamng.load_scenario(scenario)
    beamng.pause()
    beamng.start_scenario()

    vehicle.ai_drive_in_lane(True)
    vehicle.ai_set_mode("disabled")
    vehicle.ai_set_speed(10/4)

    steps = 5

    def start():
        for waypoint in waypoints[10:-1:20]:
            vehicle.ai_set_waypoint(waypoint.name)

            for idx in range(1000):

                if (idx * 0.05 * steps) > 3.:

                    sim_data_collector.collect_and_write_current_data()

                    dist = distance(sim_data_collector.last_state.pos, waypoint.position)

                    if dist < 15.0:
                        beamng.resume()
                        break

                # one step is 0.05 seconds (5/100)
                beamng.step(steps)

    try:
        start()
    finally:
        beamng.close()


if __name__ == '__main__':
    road_storage = SeedStorage('training_set')

    dataset_size = 5

    for idx in range(1, dataset_size + 1):
        path = road_storage.get_path_by_index(idx)

        road_generator = RoadGenerator(num_control_nodes=30, seg_length=20).generate()

        with open(path, 'w') as f:
            f.write(json.dumps(road_generator.to_dict()))

        street_1 = DecalRoad('street_1').add_4d_points(road_generator.sample_nodes)
        run_sim(street_1=street_1)
