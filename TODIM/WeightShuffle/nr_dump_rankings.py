import math
import numpy as np
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
data = batsmen_data
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
    original_weights = 2 * (n + 1 - ranks) / (n * (n + 1))
    return benefit_attributes, attributes, original_weights


benefit_attributes, attributes, original_weights = _initialize_items()

original_dataframe = pd.read_csv(data['scores'])
candidates = original_dataframe['Name'].to_numpy()


def get_list():
    theta = 4.0
    weights = np.copy(original_weights)
    np.random.shuffle(weights)

    raw_data = pd.DataFrame(original_dataframe, columns=attributes).to_numpy()

    dimensions = raw_data.shape
    m = dimensions[0]
    n = dimensions[1]

    for j in range(n):
        column = raw_data[:, j]
        min_val = np.min(column)
        max_val = np.max(column)
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
    # ranks = (rankdata(data) - 1).astype(int)
    # storage = np.zeros_like(candidates)
    # storage[ranks] = range(1, len(candidates) + 1)
    # return storage[::-1]
    return (len(candidates) + 1 - rankdata(data)).astype(int)


def main():
    print(f'{",".join(candidates)}')
    for _ in range(1000):
        print(
            f'{",".join(str(rank) for rank in get_list())}')


if __name__ == '__main__':
    main()
