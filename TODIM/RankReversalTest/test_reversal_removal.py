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
            c_sum = sum(column)
            raw_data[:, j] /= c_sum if c_sum != 0 else 1
        else:
            column = 1 / column
            c_sum = sum(column)
            raw_data[:, j] = column / (c_sum if c_sum != 0 else 1)

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
    print("Removed indices,Ranks Reversals")
    # # Case for count = 1
    # count = 1
    # for outer_index in range(len(name_list)):

    #     # indices = list(range(len(original_candidates)))
    #     # np.random.shuffle(indices)
    #     indices = [outer_index]

    #     names_to_skip = original_candidates[indices[:count]]
    #     skip_list = list(
    #         map(lambda player: name_list.index(player), names_to_skip))

    #     # print(skip_list)

    #     # Get the rankings for all original candidates (no skips)
    #     original_list = list(get_list([]))
    #     # Delete the ones we do not want to consider
    #     for player in names_to_skip:
    #         original_list.remove(player)

    #     processed_list = list(get_list(skip_list))

    #     mismatch_list = []

    #     for index in range(len(original_list)):
    #         expected = original_list[index]
    #         actual = processed_list[index]
    #         if expected != actual:
    #             mismatch_list.append(str(index + 1))

    #     if len(mismatch_list) > 0:
    #         print(f'"{outer_index + 1}","{",".join(mismatch_list)}"')
    count = 2
    for outer_index_i in range(len(name_list)):
        for outer_index_j in range(0, outer_index_i):

            # indices = list(range(len(original_candidates)))
            # np.random.shuffle(indices)
            indices = [outer_index_i, outer_index_j]

            names_to_skip = original_candidates[indices[:count]]
            skip_list = list(
                map(lambda player: name_list.index(player), names_to_skip))

            # print(skip_list)

            # Get the rankings for all original candidates (no skips)
            original_list = list(get_list([]))
            # Delete the ones we do not want to consider
            for player in names_to_skip:
                original_list.remove(player)

            processed_list = list(get_list(skip_list))

            mismatch_list = []

            for index in range(len(original_list)):
                expected = original_list[index]
                actual = processed_list[index]
                if expected != actual:
                    mismatch_list.append(str(index + 1))

            if len(mismatch_list) > 0:
                print(
                    f'"{outer_index_i + 1},{outer_index_j + 1}","{",".join(mismatch_list)}"')


if __name__ == '__main__':
    main()
