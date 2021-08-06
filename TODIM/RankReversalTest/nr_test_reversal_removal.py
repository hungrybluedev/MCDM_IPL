import math
import numpy as np
from numpy.core.numeric import indices
import pandas as pd
from scipy.stats import rankdata  # for ranking the candidates


bowlers_data = {
    'weights': '../../data/bowling_criteria.csv',
    'scores': '../../data/bowlers.csv',
}
batsmen_data = {
    'weights': '../../data/batting_criteria.csv',
    'scores': '../../data/batsmen.csv',
}
data = bowlers_data
attributes_data = pd.read_csv(data['weights'])


def _initialize_items():
    benefit_attributes = set()
    attributes = []
    ranks = []
    n = 0

    for i, row in attributes_data.iterrows():
        attributes.append(row['Name'])
        ranks.append(float(row['Ranking']))
        n += 1

        if row['Ideally'] == 'Higher':
            benefit_attributes.add(i)

    ranks = np.array(ranks)
    weights = 2 * (n + 1 - ranks) / (n * (n + 1))
    return benefit_attributes, attributes, weights


benefit_attributes, attributes, original_weights = _initialize_items()

original_dataframe = pd.read_csv(data['scores'])
original_candidates = original_dataframe['Name'].to_numpy()


def get_list(skip_list):
    theta = 4.0
    weights = np.copy(original_weights)
    raw_data = pd.DataFrame(original_dataframe, columns=attributes).to_numpy()
    candidates = np.delete(original_candidates, skip_list, axis=0)
    raw_data = np.delete(raw_data, skip_list, axis=0)

    dimensions = raw_data.shape
    m = dimensions[0]
    n = dimensions[1]

    for j in range(n):
        column = raw_data[:, j]
        max_val = np.max(column)
        min_val = np.min(column)
        denom = max_val - min_val
        if denom == 0:
            denom = max_val if max_val != 0 else 1
        if j in benefit_attributes:
            raw_data[:, j] = (raw_data[:, j] - min_val) / denom
        else:
            raw_data[:, j] = (max_val - raw_data[:, j]) / denom

    max_weight = max(weights)
    weights /= max_weight

    phi = np.zeros((n, m, m))

    weight_sum = sum(weights)

    for c in range(n):
        for i in range(m):
            for j in range(m):
                pic = raw_data[i, c]
                pjc = raw_data[j, c]
                val = 0
                if pic > pjc:
                    val = math.sqrt((pic - pjc) * weights[c] / weight_sum)
                if pic < pjc:
                    val = -1.0 / theta * \
                        math.sqrt(weight_sum * (pjc - pic) / weights[c])
                phi[c, i, j] = val

    delta = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            delta[i, j] = sum(phi[:, i, j])

    delta_sums = np.zeros(m)
    for i in range(m):
        delta_sums[i] = sum(delta[i, :])

    delta_min = min(delta_sums)
    delta_max = max(delta_sums)

    ratings = (delta_sums - delta_min) / (delta_max - delta_min)
    return rank_according_to(ratings, candidates)


def rank_according_to(data, candidates):
    ranks = (rankdata(data) - 1).astype(int)
    storage = np.zeros_like(candidates)
    storage[ranks] = candidates
    return storage[::-1]
    # return (len(candidates) + 1 - rankdata(data)).astype(int)


def main():
    count = 4
    # np.random.seed(120)

    indices = list(range(len(original_candidates)))
    np.random.shuffle(indices)

    names_to_skip = original_candidates[indices[:count]]
    name_list = list(original_candidates)
    skip_list = list(
        map(lambda player: name_list.index(player), names_to_skip))

    print(skip_list)

    # Get the rankings for all original candidates (no skips)
    original_list = list(get_list([]))
    # Delete the ones we do not want to consider
    for player in names_to_skip:
        original_list.remove(player)

    processed_list = list(get_list(skip_list))

    for index in range(len(original_list)):
        expected = original_list[index]
        actual = processed_list[index]
        if expected != actual:
            print(f"{index}: Expected {expected}, actually {actual}")

    print('\n')
    for index in range(len(original_list)):
        print(f'{original_list[index]}\t\t{processed_list[index]}')


if __name__ == '__main__':
    main()
