from typing import Callable

from drivebuildclient.aiExchangeMessages_pb2 import SimulationID, VehicleID
from lxml.etree import _Element

from drivebuildclient.AIExchangeService import AIExchangeService


class AI:
    def __init__(self, service: AIExchangeService) -> None:
        from core.folders import folders
        from self_driving.beamng_config import BeamNGConfig
        from self_driving.nvidia_prediction import NvidiaPrediction
        from keras.models import load_model
        self._service = service
        config = BeamNGConfig()
        model_file = str(folders.trained_models_colab.joinpath(config.keras_model_file))
        model = load_model(model_file)
        self._prediction = NvidiaPrediction(model, config)

    def start(self, sid: SimulationID, vid: VehicleID, dynamic_callback: Callable[[], None]) -> None:
        from drivebuildclient.aiExchangeMessages_pb2 import SimStateResponse, DataRequest, Control
        from self_driving.simulation_data import SimulationDataRecord
        from PIL import Image
        import io
        request = DataRequest()
        request.request_ids.append("center_cam")
        simulation_data_records = []
        while True:
            sim_state = self._service.wait_for_simulator_request(sid, vid)
            dynamic_callback()
            if sim_state == SimStateResponse.SimState.RUNNING:
                data = self._service.request_data(sid, vid, request).data
                # Request camera image
                byte_im = data["center_cam"].camera.color
                image = Image.open(io.BytesIO(byte_im)).convert("RGB")

                # Determine last state
                # is_oob, oob_counter, max_oob_percentage, oob_distance = self._oob_monitor.get_oob_info(oob_bb=False, wrt="right")
                is_oob = False
                oob_counter = max_oob_percentage = oob_distance = None
                car_state = {"timer": 0,
                             "damage": 0,
                             "pos": 0,
                             "dir": 0,
                             "vel": 0,
                             "gforces": 0,
                             "gforces2": 0,
                             "steering": 0,
                             "steering_input": 0,
                             "brake": 0,
                             "brake_input": 0,
                             "throttle": 0,
                             "throttle_input": 0,
                             "throttleFactor": 0,
                             "engineThrottle": 0,
                             "wheelspeed": 0,
                             "vel_kmh": data["ego_speed"].speed.speed
                             }
                sim_data_record = SimulationDataRecord(**car_state,
                                                       is_oob=is_oob,
                                                       oob_counter=oob_counter,
                                                       max_oob_percentage=max_oob_percentage,
                                                       oob_distance=oob_distance)
                simulation_data_records.append(sim_data_record)
                last_state = simulation_data_records[-1]
                if last_state.is_oob:
                    break

                # Compute control command
                steering_angle, throttle = self._prediction.predict(image, last_state)
                control = Control.AvCommand()
                control.steering_angle = steering_angle
                control.accelerate = throttle
                self._service.control(sid, vid, control)
            else:
                break

    @staticmethod
    def add_data_requests(ai_tag: _Element, participant_id: str) -> None:
        from lxml.etree import Element
        camera = Element("camera")
        camera.set("id", "center_cam")
        camera.set("direction", "FRONT")
        camera.set("width", "320")
        camera.set("height", "160")
        camera.set("fov", "120")
        ai_tag.append(camera)
        speed = Element("speed")
        speed.set("id", "ego_speed")
        ai_tag.append(speed)
