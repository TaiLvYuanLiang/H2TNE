import random

import numpy as np
from sklearn.manifold import TSNE

import temporal_walk
import os
import matplotlib.pyplot as plt

num_walks = 10
walk_length = 80
p = 0.2
sub_window_size = 2
alpha = 0.2
beta = 0.5


def construct_dataset_path():
    datasets_path_dict = dict()
    datasets_path_dict['dblp'] = {'data_path': './data/raw_data/dblp/',
                                  'node_types_file': './data/raw_data/dblp/node_types.txt',
                                  'rm_path': './data/rm/dblp/',
                                  'out_path': './emb/dblp/'}
    datasets_path_dict['enron'] = {'data_path': './data/raw_data/enron/',
                                   'node_types_file': './data/raw_data/enron/node_types.txt',
                                   'rm_path': './data/rm/enron/',
                                   'out_path': './emb/enron/'}
    datasets_path_dict['tky'] = {'data_path': './data/raw_data/tky/',
                                 'node_types_file': './data/raw_data/tky/node_types.txt',
                                 'rm_path': './data/rm/tky/',
                                 'out_path': './emb/tky/'}
    datasets_path_dict['movielens'] = {'data_path': './data/raw_data/movielens/',
                                       'node_types_file': './data/raw_data/movielens/node_types.txt',
                                       'rm_path': './data/rm/movielens/',
                                       'out_path': './emb/movielens/'}
    datasets_path_dict['movielens_new'] = {'data_path': './data/raw_data/movielens_new/',
                                           'node_types_file': './data/raw_data/movielens_new/node_types.txt',
                                           'rm_path': './data/rm/movielens_new/',
                                           'out_path': './emb/movielens_new/'}

    return datasets_path_dict


def sample_rm(datasets_list):
    datasets_path = construct_dataset_path()
    for dataset in datasets_list:
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                rm = temporal_walk.JustRandomWalkGenerator(datasets_path[dataset]['node_types_file'], num_walks,
                                                           walk_length, p, sub_window_size, alpha, beta)
                files = os.listdir(datasets_path[dataset]['data_path'])
                files.remove('node_types.txt')
                files.sort(key=lambda x: int(x[:-4]))
                for file in files:
                    rm_file = os.path.join(datasets_path[dataset]['rm_path'],
                                           '{0}_{1}_{2}.txt'.format(alpha, beta, file[:-4]))
                    rm.back_walk(os.path.join(datasets_path[dataset]['data_path'], file))
                    with open(rm_file, 'w+') as f:
                        for i in rm.rm_list.values():
                            for j in i.values():
                                if j.__len__() == 0:
                                    continue
                                f.write(" ".join(map(str, j)) + "\n")
                    print(rm_file)


def generate_train_sh(datasets_list, dim):
    datasets_path = construct_dataset_path()
    for dataset in datasets_list:
        # f = open('{0}_train_{1}.sh'.format(dataset, dim), 'w+')
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            f = open('{0}_{1}_{2}.sh'.format(dataset, alpha, dim), 'w+')
            for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                files = os.listdir(datasets_path[dataset]['data_path'])
                files.remove('node_types.txt')
                files.sort(key=lambda x: int(x[:-4]))
                for file in files:
                    rm_file = os.path.join(datasets_path[dataset]['rm_path'],
                                           '{0}_{1}_{2}.txt'.format(alpha, beta, file[:-4]))
                    out_file = os.path.join(datasets_path[dataset]['out_path'],
                                            '{0}_{1}_{2}.emb_{3}'.format(alpha, beta, file[:-4], dim))
                    f.write('./hhne.out -train {0} -output {1} -size {2} -thread {3}\n'.
                            format(rm_file, out_file, dim, 16))
            f.close()
        # f.close()


def generate_eval_sh(datasets_list, dim):
    datasets_path = construct_dataset_path()
    for dataset in datasets_list:
        f = open('{0}_eval_{1}.sh'.format(dataset, dim), 'w+')
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                files = os.listdir(datasets_path[dataset]['data_path'])
                files.remove('node_types.txt')
                files.sort(key=lambda x: int(x[:-4]))
                # files = [files[-1]]
                del files[-1]
                # files = ['10000.txt', '110000.txt', '210000.txt', '310000.txt', '410000.txt', '510000.txt', '555437.txt']
                for i in range(len(files)):
                    rm_file = os.path.join(datasets_path[dataset]['rm_path'],
                                           '{0}_{1}_{2}.txt'.format(alpha, beta, files[i][:-4]))
                    out_file = os.path.join(datasets_path[dataset]['out_path'],
                                            '{0}_{1}_{2}.emb_{3}.txt'.format(alpha, beta, files[i][:-4], dim))
                    f.write('python new_link_prediction.py {0} ./movielens_lp/{1}.txt 0 1\n'.
                            format(out_file, int(files[i][:-4]) + 1))
        f.close()


def degree_distribution(dataset):
    node_degree = dict()
    datasets_path = construct_dataset_path()
    raw_path = datasets_path[dataset]['data_path']
    files = os.listdir(raw_path)
    files.remove('node_types.txt')
    files.sort(key=lambda x: int(x[:-4]))
    with open(os.path.join(raw_path, files[-1]), 'r') as f:
        for line in f:
            data_line = line.strip().split('\t')
            u, v = data_line[0], data_line[2]
            if u not in node_degree:
                node_degree[u] = 0
            if v not in node_degree:
                node_degree[v] = 0
            node_degree[u] += 1
            node_degree[v] += 1
    degree_list = [0, ] * max(node_degree.values())
    for i in node_degree.keys():
        degree_list[node_degree[i] - 1] += 1
    print(sum(degree_list))
    plt.scatter(range(1, len(degree_list) + 1), degree_list, s=64)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(size=28)
    plt.yticks(size=28)
    plt.xlabel("degree of nodes", fontdict={'size': 34})
    plt.ylabel("# of nodes", fontdict={'size': 34})
    # plt.title(dataset)
    # plt.legend()
    plt.show()

    with open("tmp_distribution.csv", 'w+') as f:
        for i in range(len(degree_list)):
            f.write('{0},{1}\n'.format(i+1, degree_list[i]))



def tSNE(file_path, data_path, fig_path):
    type_dict = dict()
    color_dict = dict()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'maroon', 'orange', 'purple', 'brown', 'olive', 'gray',
              'navy', 'teal', 'lime', 'aqua', 'gold', 'silver', 'indigo', 'violet', 'tan', 'salmon', 'peru', 'pink']
    with open(data_path, 'r') as f:
        for line in f:
            data_line = line.strip().split()
            if int(data_line[0]) not in type_dict.keys():
                type_dict[int(data_line[0])] = data_line[1]
            if int(data_line[2]) not in type_dict.keys():
                type_dict[int(data_line[2])] = data_line[3]
            if data_line[1] not in color_dict.keys():
                color_dict[data_line[1]] = len(color_dict)
            if data_line[3] not in color_dict.keys():
                color_dict[data_line[3]] = len(color_dict)
    # with open('E:/2021.10动态图预测/data_pre/ml-1m/users.dat', 'r') as f:
    #     for line in f:
    #         user_id, gender, age, occu, zip = line.split('::')
    #         if int(user_id) not in type_dict.keys():
    #             type_dict[int(user_id)] = gender
    #         if gender not in color_dict.keys():
    #             color_dict[gender] = len(color_dict)
    with open(file_path, 'r') as f:
        vectors = []
        nodes = []
        for line in f:
            data_line = line.strip().split()
            if int(data_line[0]) in type_dict.keys():
                # continue
                nodes.append(data_line[0])
                vectors.append(data_line[1:])
    vectors = np.array(vectors, dtype=np.float32)
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(vectors)
    for i in range(len(Y)):
        plt.scatter(Y[i, 0], Y[i, 1], color=colors[color_dict[type_dict[int(nodes[i])]]], s=10)
    plt.xticks(())
    plt.yticks(())
    plt.axis('off')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    plt.savefig(fig_path)
    plt.show()


if __name__ == '__main__':
    # sample_rm(['movielens_new'])
    # generate_train_sh(['movielens_new'], 16)
    # generate_eval_sh(['movielens'], 128)
    degree_distribution('tky')
    # fs = []
    # i = 0
    # for _ in range(16):
    #     fs.append(open('movielens_eval_lp_128_{}.sh'.format(_), 'w+'))
    # with open('movielens_eval_128.sh', 'r') as f:
    #     for line in f:
    #         fs[i % 16].write(line)
    #         i += 1
    # for f in fs:
    #     f.close()
    # tSNE('emb/movielens/0.1_0_4.emb_128.txt', 'E:/2021.10动态图预测/data_pre/ml-1m/movie_lens_full.txt')
    # tSNE('./visual/movie128-4-4.emb', 'data/raw_data/movielens/4.txt')
    # tSNE('emb/enron/1_0.9_20.emb_16.txt', 'data/raw_data/enron/20.txt')
    # tSNE(r"E:\2021.10动态图预测\HTNE\emb\movielens\0.1_0.2_0.emb_128.txt", 'data/raw_data/movielens/4.txt', 'movie128-4-4.png')
    # dir = './data/raw_data/tky/'
    # files = os.listdir(dir)
    # files.remove('node_types.txt')
    # files.sort(key=lambda x: int(x[:-4]))
    # for file in files:
    #     fo = open('./HHNE/{0}/{1}'.format(dir.split('/')[-2], file), 'w+')
    #     with open(os.path.join(dir, file), 'r') as f:
    #         for line in f:
    #             l = line.split()
    #             fo.write('{0} {1}\n'.format(l[0], l[2]))
    #     fo.close()
    #     print(file)
