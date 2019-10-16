import json

from core.archive_impl import SmartArchive
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_problem import BeamNGProblem
from core.config import Config
from core.folder_storage import SeedStorage

config_silliest = BeamNGConfig()
config_silliest.keras_model_file = 'self-driving-car-039.h5'
config_silliest.beamng_close_at_iteration = True

config_silly = BeamNGConfig()
config_silly.keras_model_file = 'self-driving-car-065.h5'
config_silly.beamng_close_at_iteration = True

config_smart = BeamNGConfig()
config_silly.keras_model_file = 'self-driving-car-4600.h5'
config_smart.beamng_close_at_iteration = True

problem_silliest = BeamNGProblem(config_silliest, SmartArchive(config_silly.ARCHIVE_THRESHOLD))
problem_silly = BeamNGProblem(config_silly, SmartArchive(config_silly.ARCHIVE_THRESHOLD))
problem_smart = BeamNGProblem(config_smart, SmartArchive(config_smart.ARCHIVE_THRESHOLD))

# problem = BeamNGProblem(config, SmartArchive(config.ARCHIVE_THRESHOLD))

if __name__ == '__main__':
    good_members_found = 0
    attempts = 0
    storage = SeedStorage('seeds_5_silly2')

    while good_members_found < 12:
        path = storage.get_path_by_index(good_members_found + 1)
        if path.exists():
            print('member already exists', path)
            good_members_found += 1
            continue
        attempts += 1
        print(f'attempts {attempts} good {good_members_found} looking for {path}')

        member = problem_silliest.generate_random_member()
        member.evaluate()
        if member.distance_to_boundary <= 0:
            continue

        #member = problem_silly.generate_random_member()
        #member.evaluate()
        #if member.distance_to_boundary <= 0:
        #    continue
        member_smart = problem_smart.member_class().from_dict(member.to_dict())
        member_smart.config = config_smart
        member_smart.problem = problem_smart
        member_smart.clear_evaluation()
        member_smart.evaluate()
        if member_smart.distance_to_boundary <= 0:
            continue
        member.distance_to_boundary = None
        good_members_found += 1
        path.write_text(json.dumps(member.to_dict()))
