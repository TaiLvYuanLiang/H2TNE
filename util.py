import optparse
import logging
import re
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import time


def model_choice_dataset_args(usage):
    parser = optparse.OptionParser(usage)
    parser.add_option('-m', dest='m', help='Model', type='str', default='model')
    parser.add_option('-d', dest='d', help='Whole Dataset', type='str', default='foursq2014_TKY_node_format.txt')
    options, args = parser.parse_args()

    return options, args


def log_def(log_file_name="log.log"):
    import logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式化输出
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"  # 日期格式
    fp = logging.FileHandler(log_file_name, encoding='utf-8')
    fs = logging.StreamHandler()
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT,
                        handlers=[fp, fs])  # 调用handlers=[fp,fs]


def parse_model_name(model_name):
    if model_name.find(".emb") != -1:
        return 0  # .emb
    else:
        return 1  # model


def import_model(model_name):
    c = parse_model_name(model_name)
    vertex_id2vec = {}
    if c == 0:
        try:
            try:
                word_vectors = np.loadtxt(model_name, delimiter=' ')
                for line in word_vectors:
                    vertex_id2vec.update({str(int(line[0])): line[1:-1]})  # 128维
            except:
                word_vectors = KeyedVectors.load_word2vec_format(model_name)
                node_list = word_vectors.wv.vocab.keys()
                for node_id in node_list:
                    if 'a'<=node_id[0]<='z':
                        raise Exception("第三类model")
                    tmp = {node_id: word_vectors[node_id]}
                    # tmp = {node_id: list(word_vectors[node_id])}
                    vertex_id2vec.update(tmp) # 128维
        except:
            f = open(model_name)
            for line in f:
                toks = line.strip().split(' ')
                if toks.__len__() < 3:
                    continue
                tmp = {toks[0][1:]: list(map(float, toks[1:]))}
                vertex_id2vec.update(tmp)
            f.close()

    else:
        model = Word2Vec.load(model_name)
        word_vectors = KeyedVectors.load(model_name)

        for key in word_vectors.wv.vocab.keys():
            vertex_id2vec.update({key: model.wv.__getitem__(key)})
    return vertex_id2vec  # str->list of int


def import_node_label_dict(dsname):
    node_data = np.loadtxt(dsname, delimiter="\t").tolist()
    node_data_dict = {}
    for edge in node_data:
        node_data_dict.update({str(int(edge[0])): str(int(edge[1]))})
        node_data_dict.update({str(int(edge[2])): str(int(edge[3]))})
    return node_data_dict


def import_net(net_path):
    """
    import dataset from net_path
    :return:
    """
    logging.info("start input dataset")
    edge_dict = {}
    time_dict = {}
    io_cost = 0
    try:
        import_net_start = time.process_time()
        all_edge_list = np.loadtxt(net_path, delimiter='\t')
        all_edge_list = all_edge_list.astype(np.int64)
        for edge in all_edge_list:
            if edge[0] == edge[1]:
                continue
            back_node_accord_edge = edge_dict.get(edge[1])
            front_node_accord_edge = edge_dict.get(edge[0])

            if back_node_accord_edge is None:
                back_node_accord_edge = [[edge[0], edge[2]]]
            else:
                back_node_accord_edge.append([edge[0], edge[2]])

            edge_dict.update({edge[1]: back_node_accord_edge})

            if front_node_accord_edge is None:
                front_node_accord_edge = [[edge[1], edge[2]]]
            else:
                front_node_accord_edge.append([edge[1], edge[2]])

            edge_dict.update({edge[0]: front_node_accord_edge})

            time_accord_edge = time_dict.get(edge[2])

            if time_accord_edge is None:
                time_accord_edge = [edge[0:2].tolist(),
                                    [edge[1], edge[0]]]
            else:
                time_accord_edge.append(edge[0:2].tolist())
                time_accord_edge.append([edge[1], edge[0]])
            time_dict.update({edge[2]: time_accord_edge})

        import_net_end = time.process_time()
        io_cost = io_cost + (import_net_end - import_net_start)
        logging.info("finish input dataset")
        return edge_dict, time_dict, io_cost
    except Exception as e:
        logging.error("Load dataset error!")
        print(e)
        return edge_dict, time_dict, io_cost


def import_edges(path, line_len='5'):
    edges_data = np.loadtxt(path, delimiter="\t").astype(np.int).tolist()
    pos_sample = []
    nodes = []
    first = 0
    second = 2 if line_len == '5' else 1
    for edges in edges_data:
        pos_sample.append([str(edges[first]), str(edges[second])])
        nodes.append(str(edges[first]))
        nodes.append(str(edges[second]))

    return pos_sample, set(nodes)
