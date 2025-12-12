# sisa_unlearning_experiment
This project is a benchmark testing for SISA experiments with different levels leakage

# File Structure & Process

### data_split_strategy.ipynb

This file takes in the raw image data and shards them with a bias aligning a leakage strategy.

Each shard has a dominant class animal with 800 images + leakage from other class of 50 each. 

The dominant class animal's images are also leaked into other shards at 50 images in each shard

There is also option to limit the leakage by only allowing leakage to happen in a neighboring shard and removing all leakage in the dataset for a high/low leakage experiment.

### sisa_unlearning.ipynb

This file loads the shards created by data_split_strategy and loads them into a DataLoader that will serve the ML models with batches (splicing in SISA). After the images are loaded with either 5 or 3 classes, SISA unlearning is performed inside the notebook and saves sharded models into the my_models folder. 

Unlearning is provided for the "cat" class, where the majority of the class images are removed from the dominant class and the corresponding model is dropped. Then it is retrained with the "cat" class removed from the shard and a new model replaces the original model in the model esemble. 

There is also a block to train the model in the traditional approach, with the full dataset that does not allow for unlearning, unless the entire model is retrained.

### evaluate.ipynb

This is the template for the results of the experiment.

For the experiment, we conducted two approaches;

1. Three classes (cat, dog, horse)
2. Five classes (cat, dog, horse, chicken, elephant)

Under each corresponding folder of the approach, there is a further breakdown of two scenarios.

1. High Leakage: the data we're trying to delete (class: cat) is founded across all shards and have leakage in every model shards trained.
2. Low Leakage: the "cat" class is only present in the neighboring class and there is no leakage in other shards.

Each of the scenario folders has its own evaluate.ipynb, where one can load the models and explore the effects of SISA; with stats provided before deletion, after deletion of class requested, and post retraining of effected shard.