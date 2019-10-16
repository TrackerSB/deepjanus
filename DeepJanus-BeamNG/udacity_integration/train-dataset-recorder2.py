import random
from typing import Tuple, List

import numpy as np

from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.decal_road import DecalRoad
from .training_data_collector_and_writer import TrainingDataCollectorAndWriter
from self_driving.utils import get_node_coords, points_distance

random.seed(42)
np.random.seed(42)



def run_sim(nodes):
    print(nodes)
    brewer = BeamNGBrewer(road_nodes=nodes)
    beamng = brewer.beamng
    waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))
    maps.install_map_if_needed()
    maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())
    vehicle = brewer.setup_vehicle()
    brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()
    #camera = brewer.setup_scenario_camera()
    street_1 = brewer.decal_road
    sim_data_collector = TrainingDataCollectorAndWriter(vehicle, beamng, street_1)
    brewer.bring_up()

    steps = 5

    def start():
        for idx in range(1000):
            if (idx * 0.05 * steps) > 3.:
                sim_data_collector.collect_and_write_current_data()
                dist = points_distance(sim_data_collector.last_state.pos, waypoint_goal.position)
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

    # sin
    points: List[Tuple[float, float]] = [[x, np.sin(x / 10) * 10] for x in np.arange(0., 900., 2.)]
    nodes = [[x, 0., -28., 8.] for x in np.arange(0., 900., 2.)]

    run_sim(nodes)
