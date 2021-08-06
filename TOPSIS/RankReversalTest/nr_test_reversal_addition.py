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
original_candidates = original_dataframe['Name'].to_numpy()


def get_list(skip_list):
    weights = np.copy(original_weights)
    raw_data = pd.DataFrame(original_dataframe, columns=attributes).to_numpy()
    candidates = np.delete(original_candidates, skip_list, axis=0)
    raw_data = np.delete(raw_data, skip_list, axis=0)

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

    raw_data *= weights
    a_pos = np.copy(weights)
    a_neg = np.zeros(n)

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    cs_order = rank_according_to(cs, candidates)
    # sp_order = rank_according_to(sp)
    # sn_order = rank_according_to(sn)
    return cs_order


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
