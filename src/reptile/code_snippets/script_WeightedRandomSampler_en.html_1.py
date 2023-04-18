from paddle.io import WeightedRandomSampler

sampler = WeightedRandomSampler(weights=[0.1, 0.3, 0.5, 0.7, 0.2],
                                num_samples=5,
                                replacement=True)

for index in sampler:
    print(index)