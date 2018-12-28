"""Define a genetic meta-learning algorithm for parameter initialization search"""
from absl import flags
from copy import deepcopy
from math import ceil
import numpy as np

FLAGS = flags.FLAGS


class GeneticAlgorithm(object):
    """Implements Baldwinian genetic meta-learning algorithm"""

    def __init__(self,
                 sampling,
                 hparams,
                 meta_data,
                 population_size,
                 num_children,
                 best_sample,
                 lucky_few,
                 chance_of_mutation,
                 fraction_to_mutate):
        """Initializes genetic algorithm. Sets the initial population to the
           randomly chosen values around the default
        Args:
            sampling: Sampling for which the initialization is to be learned
            hparams: Hyperparameters to the sampling algorithm
            meta_data: List of BanditDataset objects used for meta-learning
            population_size: Size of the population
            num_children: Number of children per pair of parents
            best_sample: Number of top fitted individuals to use for reproduction
            lucky_few: Number of random individuals to use for reproduction
            chance_of_mutation: Chance of mutating each parameter entry
            fraction_to_mutate: Fraction of parameter entries to mutate
        """

        if (best_sample + lucky_few) / 2 * num_children != population_size:
            raise ValueError("Population size not stable.")

        self.sampling = sampling
        self.hparams = hparams
        self.meta_data = meta_data

        self.population_size = population_size
        self.num_children = num_children
        self.best_sample = best_sample
        self.lucky_few = lucky_few
        self.chance_of_mutation = chance_of_mutation
        self.fraction_to_mutate = fraction_to_mutate

        self.population = self.initialize_population()

    def add_randomness(self, parameters):
        randomized = deepcopy(parameters)
        for k, v in randomized.items():
            array, lb, ub = v
            array += lb + np.random.rand(*array.shape) * (ub - lb)
        return randomized

    def initialize_population(self):
        instance = self.sampling(self.hparams, None)
        default_parameters = instance.trainable_parameters()

        population = []
        for i in range(self.population_size):
            population.append(self.add_randomness(default_parameters))

        return population

    def meta_train(self, params):
        fitness = 0
        precision = 0
        recall = 0
        for data in self.meta_data:
            instance = self.sampling(self.hparams, data)
            instance.reset_trainable_parameters(params)
            summary = instance.run()
            fitness += -summary.cost[-1]
            precision += summary.precision[-1]
            recall += summary.recall[-1]

        return fitness / len(self.meta_data), precision / len(self.meta_data), recall / len(self.meta_data)

    def compute_population_fitness(self):
        population = []
        for i, params in enumerate(self.population):
            fitness, precision, recall = self.meta_train(params)
            if FLAGS.meta_verbose:
                print('\t\tTrained individual {:3}.'
                      'Fitness: {:20} | Precision: {:20} | Recall: {:20}'.format(i + 1, fitness, precision, recall))
            population.append((fitness, params))

        return sorted(population, key=lambda x: x[0], reverse=True)

    def select_parents(self, sorted_population):
        parents = []
        for i in range(self.best_sample):
            parents.append(sorted_population[i][1])
        for i in range(self.lucky_few):
            parents.append(sorted_population[np.random.choice(len(sorted_population))][1])
        np.random.shuffle(parents)
        return parents

    def breed(self, parent_1, parent_2):
        child = {}
        for k, v in parent_1.items():
            array1, l1, u1 = v
            array2, l2, u2 = parent_2[k]

            if l1 != l2 or u1 != u2:
                raise ValueError('Different bounds on same parameter')

            child_array = np.zeros(shape=array1.shape)
            mask = np.random.randint(2, size=array1.shape).astype(bool)
            child_array[mask] = array1[mask]
            child_array[np.logical_not(mask)] = array2[np.logical_not(mask)]
            child[k] = (child_array, l1, u1)
        return child

    def reproduce(self, parents):
        children = []
        for i in range(int(len(parents) / 2)):
            for _ in range(self.num_children):
                children.append(self.breed(parents[i], parents[-i - 1]))
        return children

    def mutate_child(self, child):
        for k, v in child.items():
            array, lb, ub = v
            num_entries_to_mutate = ceil(np.size(array) * self.fraction_to_mutate / 100)
            entries_to_mutate = np.random.choice(np.size(array),
                                                 size=num_entries_to_mutate,
                                                 replace=False)
            param_lin = array.reshape(-1)
            param_lin[entries_to_mutate] += lb + np.random.rand(num_entries_to_mutate) * (ub - lb)

        return child

    def mutate(self, children):
        for i in range(len(children)):
            if np.random.random() * 100 < self.chance_of_mutation:
                children[i] = self.mutate_child(children[i])
        return children

    def next_generation(self):
        if FLAGS.meta_verbose:
            print('\tComputing population fitness')
        sorted_population = self.compute_population_fitness()
        if FLAGS.meta_verbose:
            print('\tCurrent best fitness:', sorted_population[0][0])
            print('\tSelecting parents')
        parents = self.select_parents(sorted_population)
        if FLAGS.meta_verbose:
            print('\tReproducing')
        children = self.reproduce(parents)
        if FLAGS.meta_verbose:
            print('\tMutating')
        next_generation = self.mutate(children)

        return next_generation

    def run(self, num_generations):
        if FLAGS.meta_verbose:
            print('Running genetic meta-learning for {} generations'.format(num_generations))
        for i in range(num_generations):
            if FLAGS.meta_verbose:
                print('>> Reproducing generation {}\n'.format(i + 1))
            self.population = self.next_generation()
            if FLAGS.meta_verbose:
                print()

    def get_best_individual_from_population(self):
        return self.compute_population_fitness()[0]
