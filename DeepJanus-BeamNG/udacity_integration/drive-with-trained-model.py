import os
import random

import numpy as np

import udacity_utils as utils
from beamng_brewer import BeamNGBrewer
from beamng_car_cameras import BeamNGCarCameras
from beamng_waypoint import BeamNGWaypoint
from core.folders import folders
from road_storage import RoadStorage
from simulation_data import SimulationDataRecord, SimulationData
from simulation_data_collector import SimulationDataCollector
from beamng_tig_maps import maps
from vehicle_state_reader import VehicleStateReader

maps.install_map_if_needed()

random.seed(42)
np.random.seed(42)


def get_node_coords(node):
    return node[0], node[1], node[2]


def distance(p1, p2):
    return np.linalg.norm(np.subtract(get_node_coords(p1), get_node_coords(p2)))


class Prediction:
    def __init__(self, model, max_speed):
        self.model = model
        self.max_speed = max_speed
        self.speed_limit = max_speed

    def predict(self, image, car_state: SimulationDataRecord):
        try:
            image = np.asarray(image)

            image = utils.preprocess(image)
            image = np.array([image])

            steering_angle = float(self.model.predict(image, batch_size=1))

            speed = car_state.vel_kmh
            if speed > self.speed_limit:
                self.speed_limit = 10  # slow down
            else:
                self.speed_limit = self.max_speed
            throttle = 1.0 - steering_angle ** 2 - (speed / self.speed_limit) ** 2
            return steering_angle, throttle

        except Exception as e:
            print(e)


def run_sim(nodes, model_file, sim_name, speed_limit) -> SimulationDataCollector:
    if not os.path.exists(model_file):
        raise Exception(f'File {model_file} does not exist!')

    brewer = BeamNGBrewer(road_nodes=nodes)
    beamng = brewer.beamng
    waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))
    maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())
    vehicle = brewer.setup_vehicle()
    camera = brewer.setup_scenario_camera()

    cameras = BeamNGCarCameras()
    vehicle_state_reader = VehicleStateReader(vehicle, beamng, additional_sensors=cameras.cameras_array)
    brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

    steps = brewer.params.beamng_steps
    sim_data_collector = SimulationDataCollector(vehicle, beamng, brewer.decal_road, brewer.params,
                                                 vehicle_state_reader=vehicle_state_reader,
                                                 camera=camera,
                                                 simulation_name=sim_name)

    brewer.bring_up()

    from keras.models import load_model
    model = load_model(model_file)
    predict = Prediction(model, speed_limit)

    def start():
        for _ in range(1000):
            sim_data_collector.collect_current_data(oob_bb=False)
            last_state: SimulationDataRecord = sim_data_collector.states[-1]
            if distance(last_state.pos, waypoint_goal.position) < 25.0:
                break

            if last_state.is_oob:
                sim_data_collector.take_car_picture_if_needed()
                break
            img = vehicle_state_reader.sensors['cam_center']['colour'].convert('RGB')
            steering_angle, throttle = predict.predict(img, last_state)
            vehicle.control(throttle=throttle, steering=steering_angle, brake=0)
            beamng.step(steps)

    try:
        start()
        return sim_data_collector
    finally:
        sim_data_collector.save()
        beamng.close()


if __name__ == '__main__':
    # Create here the nodes, so that both simulation use the same road.

    # cache_filename = 'cached_road_nodes.json'
    # if os.path.exists(cache_filename):
    #     with open(cache_filename, 'r') as f:
    #         nodes = json.loads(f.read())
    # else:
    #     nodes = RoadGenerator.generate(num=40, max_angle=360, visualise=False, verbose=False)
    #     with open(cache_filename, 'w') as f:
    #         f.write(json.dumps(nodes))

    # nodes = [[x, np.sin(x / 15) * 10, -28., 8.] for x in np.arange(0., 900., 2.)]
    # nodes = [[x, 0, -28., 8.] for x in np.arange(0., 900., 2.)]
    models = {
        'm65': ('self-driving-car-065.h5', 'm65', 25.0),
        'm4600': ('self-driving-car-4600.h5', 'm4600', 25.0),
        'test1': ('self-driving-car-4600.h5', 'test1', 25.0),
    }
    model_name, sim_folder, speed_limit = models['m4600']
    model_path = str(folders.trained_models_colab.joinpath(model_name))

    sim = SimulationData('beamng_nvidia_runner/sim_2019-08-28--11-06-40').load()
    run_sim(sim.road.nodes, model_path, sim_name='man1/1', speed_limit=speed_limit)

    road_storage = RoadStorage()
    for round_idx in range(10):
        for road_idx in range(11, 21):
            sim_name = f'{sim_folder}/sim_road{road_idx}_round{round_idx}'
            print(f'simulation {sim_name}', end='')
            simulation_data = SimulationData(sim_name)
            if simulation_data.complete():
                print(' already exists')
            else:
                print(' starting now')
                run_sim(RoadStorage().get_road_nodes_by_index(road_idx)
                        , model_path
                        , sim_name=sim_name
                        , speed_limit=speed_limit)
