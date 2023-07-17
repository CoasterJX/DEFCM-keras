import numpy as np
import itertools
import sys
sys.path.append('../src')

from DEFCM import DEFCM
from datasets import load_data
from keras.initializers import VarianceScaling
from keras.optimizers import SGD

from update_summary import generate_summary_results


EXP_GROUP = {
    'test_data': ['mnist'],
    # 'fuzzifier': [round(x, 1) for x in np.arange(1.1, 1.7, 0.1)],
    'trials': list(range(1)),
    'learning_rate': [1e-1],
    # 'M': [1.1, 1.3, 1.5, 1.7, 1.9, 2, 3, 4, 5, 6, 7, 8],
    # 'norm_factor': [2],
    
    'saved_cluster_epochs': [0, 3, 5, 10, 15, 20, 25, 30, 50],
    'summary_config': ('facc', max),
    'new_version': True,
    'idefcm': True
}


def run_one_experiment_group(exp_group):

    optimum_fuzzifier_checklist = {
        ('mnist', 'fmnist'): 1.5,
        ('stl', 'usps', 'reuters'): 1.3
    }

    if 'n_clusters' not in exp_group:
        exp_group['n_clusters'] = [-1]
    if 'fuzzifier' not in exp_group:
        exp_group['fuzzifier'] = [-1]
    if 'learning_rate' not in exp_group:
        exp_group['learning_rate'] = [0.01]
    if 'M' not in exp_group:
        exp_group['M'] = [2]
    if 'norm_factor' not in exp_group:
        exp_group['norm_factor'] = [2]
    if 'trials' not in exp_group:
        exp_group['trials'] = ['_testonly']

    data_dict = {}
    for td in exp_group['test_data']:
        data_dict[td] = load_data(td)

    for test_data, n_clusters, fuzzifier, trial, learning_rate, m, norm_factor in itertools.product(
        exp_group['test_data'],
        exp_group['n_clusters'],
        exp_group['fuzzifier'],
        exp_group['trials'],
        exp_group['learning_rate'],
        exp_group['M'],
        exp_group['norm_factor']
    ):

        x, y = data_dict[test_data]
        if n_clusters == -1:
            n_clusters = len(np.unique(y))
        if fuzzifier == -1:
            for k, v in optimum_fuzzifier_checklist.items():
                if test_data in k:
                    fuzzifier = v
                    break
            if fuzzifier == -1:
                raise ValueError(f'Optimum fuzzifier undefined for dataset: {test_data}')

        defcm = DEFCM(
            [x.shape[-1], 500, 500, 2000, 10],
            n_clusters=n_clusters,
            fuzzifier=fuzzifier,
            m=m,
            norm_factor=norm_factor,
            init=VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform') if test_data in ['mnist', 'fmnist'] else 'glorot_uniform',
            exp_name=test_data,
            new_version=exp_group.get('new_version', True),
            improved=exp_group.get('idefcm', True),
            cluster_csv_epochs=exp_group['saved_cluster_epochs']
        )

        defcm.pretrain(
            x,
            y=y,
            optimizer=SGD(lr=1, momentum=0.9) if test_data in ['mnist', 'fmnist'] else 'adam',
            epochs=300 if test_data in ['mnist', 'fmnist'] else 50,
            batch_size=256
        )

        defcm.exp_name = f'''
            {defcm.exp_name}
            /nclu={n_clusters}
            /fuzzifier={fuzzifier}
            /learning_rate={learning_rate}
            /m={m}
            /norm_factor={norm_factor}
            /trial{trial}'''.replace(' ', '').replace('\n', '')
        defcm.compile(optimizer=SGD(learning_rate=learning_rate, momentum=0.9), loss='kld')
        defcm.fit(
            x,
            y=y,
            max_iter=2e4,
            batch_size=256,
            tol=1e-3,
            update_interval=140 if test_data in ['mnist', 'fmnist'] else 30
        )


if __name__ == '__main__':

    # with open(f'''../log/{"+".join(EXP_GROUP["test_data"])}
    #     nclu={"~".join(EXP_GROUP.get("n_clusters", ["#"]))}
    #     fuzzifier={"~".join(EXP_GROUP.get("fuzzifier", ["#"]))}
    #     trials={len(EXP_GROUP.get("trials", [1]))}
    #     save_clusters={EXP_GROUP.get("saved_cluster_epochs", [])}.log'''
    #           .replace(' ', '')
    #           .replace('\n', ' '), 'w') as sys.stdout:

    run_one_experiment_group(EXP_GROUP)

    generate_summary_results(
        '../data/exp_result_DEFCM/',
        optimum_standard=EXP_GROUP['summary_config'][0],
        optimizer=EXP_GROUP['summary_config'][1])
