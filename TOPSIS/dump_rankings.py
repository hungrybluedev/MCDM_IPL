import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata  # for ranking the candidates


bowlers_data = {
    'weights': '../data/bowling_criteria.csv',
    'scores': '../data/bowlers.csv',
}
batsmen_data = {
    'weights': '../data/batting_criteria.csv',
    'scores': '../data/batsmen.csv',
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
    weights = np.copy(original_weights)
    np.random.shuffle(weights)

    raw_data = pd.DataFrame(original_dataframe, columns=attributes).to_numpy()

    dimensions = raw_data.shape
    m = dimensions[0]
    n = dimensions[1]

    divisors = np.empty(n)
    for j in range(n):
        column = raw_data[:, j]
        divisors[j] = np.sqrt(column @ column)

    raw_data /= divisors
    raw_data *= weights
    a_pos = np.zeros(n)
    a_neg = np.zeros(n)
    for j in range(n):
        column = raw_data[:, j]
        max_val = np.max(column)
        min_val = np.min(column)

        # See if we want to maximize benefit or minimize cost (for PIS)
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    cs_order = rank_according_to(cs)
    # sp_order = rank_according_to(sp)
    # sn_order = rank_according_to(sn)
    return cs_order


def rank_according_to(data):
    # ranks = (rankdata(data) - 1).astype(int)
    # storage = np.zeros_like(candidates)
    # storage[ranks] = candidates
    # return storage[::-1]
    return (len(candidates) + 1 - rankdata(data)).astype(int)


def main():
    print(f'{",".join(candidates)}')
    for _ in range(1000):
        print(','.join(str(rank) for rank in get_list()))


if __name__ == '__main__':
    main()
