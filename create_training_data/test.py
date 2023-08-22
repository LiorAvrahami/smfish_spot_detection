import syspath
syspath.append_git_root()
from create_training_data.training_data_generator import TrainingDataGenerator
a = TrainingDataGenerator.make_default_training_data_generator(3)

images,tags = a.get_next_batch()