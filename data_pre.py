import pandas as pd

path = r"data_pre/"

def create_graph():
    df = pd.read_csv(path + r'\ml-1m\ratings.dat', sep='::', header=None, engine='python')
    df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    movies = dict()
    with open(path + r'\ml-1m\movies.dat', 'r', encoding='Windows 1252') as f:
        for line in f:
            print(line)
            movie_id, title, genres = line.split('::')
            movies[movie_id] = genres.strip().split('|')[0]
    min_ts = df['timestamp'].min()
    f = open(path + r'\ml-1m\movie_lens_full.txt', 'w+')
    for i in range(len(df)):
        user_id = str(df['user_id'][i])
        movie_id = str(df['movie_id'][i])
        genres = movies[movie_id]
        timestamp = df['timestamp'][i] - min_ts
        f.write(str(user_id) + '\tuser\t' + str(movie_id) + '\t' + str(genres) + '\t' + str(timestamp) + '\n')
    f.close()


def devide_ts():
    df = pd.read_csv(path + r'\ml-1m\movie_lens_full.txt', sep='\t', header=None, engine='python')
    df.columns = ['user_id', 'user_type', 'movie_id', 'genres', 'timestamp']
    df['timestamp'] = (df['timestamp'].astype(int) / 1000).astype(int)
    df['genres'] = 'movie'
    # max = df['timestamp'].max()
    df.sort_values(by=['timestamp'], inplace=True)
    for i in range(5):
        df_temp = df.head(df.shape[0] * (i + 1) // 5).tail(df.shape[0] // 5)
        df_temp['user_type'] = 1
        df_temp['genres'] = 2
        df_temp.to_csv(r'./lp/{}.txt'.format(i), sep='\t', header=None, index=False)


if __name__ == '__main__':
    # create_graph()
    # # devide_ts()
    # list = ['./data/rm/movielens/1_1_0.txt', './data/rm/movielens/1_1_1.txt',
    #         './data/rm/movielens/1_1_2.txt', './data/rm/movielens/1_1_3.txt',
    #         './data/rm/movielens/1_1_4.txt']
    # import random
    # for p in list:
    #     with open(p, 'r') as f:
    #         lines = []
    #         for line in f:
    #             lines.append(line)
    #         random.shuffle(lines)
    #         with open('LIME/' + p[24:25] + '/' + p[24:25] + '.train.txt', 'w+') as f1:
    #             f1.writelines(lines[:int(len(lines) * 0.8)])
    #         with open('LIME/' + p[24:25] + '/' + p[24:25] + '.test.txt', 'w+') as f2:
    #             f2.writelines(lines[int(len(lines) * 0.8):int(len(lines) * 0.9)])
    #         with open('LIME/' + p[24:25] + '/' + p[24:25] + '.valid.txt', 'w+') as f3:
    #             f3.writelines(lines[int(len(lines) * 0.9):])
    # import networkx as nx
    # import numpy as np
    # x = []
    # id2idx = dict()
    # for i in range(5):
    #     with open('./data/raw_data/movielens/{}.txt'.format(i), 'r') as f:
    #         g = nx.Graph()
    #         for line in f:
    #             line_list = line.split('\t')
    #             u, v = line_list[0], line_list[2]
    #             if u not in id2idx:
    #                 id2idx[u] = len(id2idx)
    #             if v not in id2idx:
    #                 id2idx[v] = len(id2idx)
    #             g.add_edge(id2idx[u], id2idx[v])
    #         x.append(g)
    # graphs = np.array(x)
    # np.savez('graphs.npz', graph=graphs)
    # with open('id_dict.txt', 'w+') as f:
    #     for k, v in id2idx.items():
    #         f.write(k + '\t' + str(v) + '\n')
    import pickle
    emb = pickle.load(open('./movielens_full_4.emb', 'rb'))
    with open('GCN.emb', 'w+') as f:
        for node in emb.keys():
            f.write(str(node) + ' ')
            for i in emb[node]:
                f.write(str(i) + ' ')
            f.write('\n')
