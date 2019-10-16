import random
from typing import Tuple, List

import numpy as np

from beamng_tig_maps import maps
from beamng_waypoint import BeamNGWaypoint
from beamngpy import BeamNGpy, Scenario, Vehicle
from decal_road import DecalRoad
from road_points import RoadPoints
from training_data_collector_and_writer import TrainingDataCollectorAndWriter

maps.install_map_if_needed()

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


def run_sim(nodes, ai_aggression, street_1: DecalRoad, street_2: DecalRoad):
    waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(street_1.nodes[-1]))

    maps.beamng_map.generated().write_items(
        street_1.to_json() + '\n' + waypoint_goal.to_json() + '\n' + street_2.to_json())

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

    vehicle.ai_set_aggression(ai_aggression)
    vehicle.ai_drive_in_lane(True)
    # vehicle.ai_set_speed(25.0 / 4)
    vehicle.ai_set_waypoint(waypoint_goal.name)
    # vehicle.ai_set_mode("manual")

    # sleep(5)

    steps = 5

    print(nodes)
    print(beamng.get_road_edges("street_1"))

    def start():
        for idx in range(1000):
            if (idx * 0.05 * steps) > 3.:
                sim_data_collector.collect_and_write_current_data()
                dist = distance(sim_data_collector.last_state.pos, waypoint_goal.position)
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
    # Create here the nodes, so that both simulation use the same road.
    # cache_filename = 'cached_road_nodes.json'
    # if os.path.exists(cache_filename):
    #     with open(cache_filename, 'r') as f:
    #         nodes = json.loads(f.read())
    # else:
    #     nodes = RoadGenerator(num_control_nodes=5,seg_length=20).generate()
    #     with open(cache_filename, 'w') as f:
    #         f.write(json.dumps(nodes))

    points: List[Tuple[float, float]] = [[x, np.sin(x / 10) * 10] for x in np.arange(0., 900., 2.)]
    nodes = [[x, 0, -28., 8.] for x in np.arange(0., 900., 2.)]

    street_1 = DecalRoad('street_1', drivability=1, material='').add_2d_points(points, width=4)
    dummy = DecalRoad('dummy').add_2d_points(points, width=6)

    street_2 = DecalRoad('street_2', drivability=0).add_2d_points(RoadPoints.from_nodes(dummy.nodes).left, width=12)

    run_sim(nodes, ai_aggression=0.5, street_1=street_1, street_2=street_2)
