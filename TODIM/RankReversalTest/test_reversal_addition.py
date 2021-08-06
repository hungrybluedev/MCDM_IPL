import math
import numpy as np
from numpy.core.numeric import indices
import pandas as pd
from scipy.stats import rankdata  # for ranking the candidates


bowlers_data = {
    'weights': '../../data/bowling_criteria.csv',
    'scores': '../../data/bowlers_extra.csv',
}
batsmen_data = {
    'weights': '../../data/batting_criteria.csv',
    'scores': '../../data/batsmen_extra.csv',
}
data = batsmen_data
# data = bowlers_data
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
        if j in benefit_attributes:
            raw_data[:, j] /= sum(column)
        else:
            column = 1 / column
            raw_data[:, j] = column / sum(column)

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
    name_list = list(original_candidates)
    print('Index Added,Rank Reversals')
    for location, new_player in enumerate(name_list):
        original_list = list(get_list([location]))

        processed_list = list(get_list([]))
        processed_list.remove(new_player)

        mismatch_list = []

        for index in range(len(original_list)):
            expected = original_list[index]
            actual = processed_list[index]
            if expected != actual:
                mismatch_list.append(str(index + 1))

        if len(mismatch_list) > 0:
            print(f'"{location + 1}","{",".join(mismatch_list)}"')


if __name__ == '__main__':
    main()
