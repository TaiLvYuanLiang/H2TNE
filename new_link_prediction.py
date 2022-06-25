#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @file   : new_link_prediction.py
# @author : Zhutian Lin
# @date   : 2019/10/17
# @version: 1.0
# @desc   : 完成链路预测任务
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"                        # 日期格式
fp = logging.FileHandler('config.log', encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fs, fp])    # 调用handlers=[fp,fs]
#  输出到命令行中
#  计时器
import time
import sys


class link_predictor:
    # dataset_name = "E:\MLcode\DBIS\\new_link_prediction\\test.txt"

    def __init__(self):
        logging.info("初始化分类器")

        self.dataset = []  # 读入的真实数据（list）
        self.vectors_dict = {}  # 每个点对应的字典{点index(str):向量(nparray)} 这个str可不能带小数点
        self.legal_edge = []  # 要求每个点的index组成的node1，node2构成的list的list
        self.test_edge = []
        self.test_label = []  # 每个边对的label
        self.test_cosine_distance = []  # 每个边的余弦距离
        self.node_list = []
        self.old_test_length = 0
        self.new_test_length = 0
    # 操作：首先导入数据集，格式不变；原来的test_set输入为以空格为分隔符的node1 node2，主程序的embedding，格式不变

    def import_dataset(self,dataset_name,option=0):

        self.dataset = np.loadtxt(dataset_name, delimiter='\t')  # 读入的真实数据
        # self.dataset = np.loadtxt(dataset_name, delimiter=',')  # 读入的真实数据
        if option == 0:  # 认为是有label的
            self.legal_edge = [[node1, node2] for (node1, node2, label, time) in self.dataset]
        else:
            self.legal_edge = [[node1, node2] for (node1, node2, time) in self.dataset]
        self.legal_edge = np.array(self.legal_edge)
        self.legal_edge = self.legal_edge.astype(np.int32)  # 去掉小数点
        self.legal_edge = self.legal_edge.astype(np.str)  # 转换成和键值一样的情况（长得是跟int一样的str，后面没有小数点）
        self.legal_edge = self.legal_edge.tolist()

    def import_model(self, model_name, option=0):
        #  0代表主方法，model_name代表的是主函数输出的节点向量
        if option == 0:
            word_vectors = np.loadtxt(model_name, delimiter=' ')
            for line in word_vectors:
                tmp = {str(int(line[0])): line[1:-1]}
                self.vectors_dict.update(tmp)
            self.node_list = [node for node in self.vectors_dict]

        #  1代表online，model_name代表的是online保存的sg模型
        if option == 1:
            model = Word2Vec.load(model_name)
            word_vectors = KeyedVectors.load(model_name)
            # 构造新字典
            for key in word_vectors.wv.vocab.keys():
                tmp = {key: model.wv.__getitem__(key)}
                self.vectors_dict.update(tmp)
            self.node_list = [node for node in self.vectors_dict]

        #  2代表使用的是标准sg，model_name代表的是标准sg保留的向量（首行为节点数和维度大小）
        if option == 2:
            word_vectors = KeyedVectors.load_word2vec_format(model_name)
            self.node_list = list(word_vectors.wv.vocab.keys())
            for node_id in self.node_list:
                tmp = {node_id: word_vectors[node_id]}
                self.vectors_dict.update(tmp)

        #  3代表自己生成的向量，节点第一位为字母表示
        if option == 3:
            f = open(model_name)
            for line in f:
                toks = line.strip().split(' ')
                if toks.__len__() < 3:
                    continue
                tmp = {toks[0][1:]: list(map(float, toks[1:]))}
                self.vectors_dict.update(tmp)
            self.node_list = [node for node in self.vectors_dict]
            f.close()

    def get_test_set(self, test_name, option=1):
        logging.info("进入获取训练集阶段")
        #  输入为以空格为分隔符的node1 node2

        test = np.loadtxt(test_name, delimiter='\t')  # 读入的真实数据
        # test = np.loadtxt(test_name, delimiter=',')  # 读入的真实数据
        # if option == 0:
        #     self.test_edge = [[node1, node2] for (node1, node2, label, time) in test]
        # else:
        #     self.test_edge = [[node1, node2] for (node1, node2, label) in test]
        if option == 1:
            self.test_edge = [[node1, node2] for (node1, label1, node2, label2, e_time) in test]
        else:
            self.test_edge = [[node1, node2] for (node1, node2, e_time) in test]

        self.test_edge = np.array(self.test_edge)
        self.test_edge = self.test_edge.astype(np.int32)  # 去掉小数点
        self.test_edge = self.test_edge.astype(np.str)  # 转换成和键值一样的情况（长得是跟int一样的str，后面没有小数点）
        self.test_edge = self.test_edge.tolist()
        #  面对删去元素的情况，可能会出现删掉之后索引变乱了，比如删了14，下一个元素挪到14，可是已经遍历到15了，14就没检查了
        self.old_test_length = len(self.test_edge)
        self.test_edge = [edge for edge in self.test_edge if edge[0] in self.vectors_dict and edge[1] in self.vectors_dict]
        self.new_test_length = len(self.test_edge)
        self.test_label = np.ones(self.new_test_length).tolist()

        for i in range(self.new_test_length):
            illegal_edge_candi = self.get_illegal_edge()
            if i%int(self.new_test_length/10) == 0:
                logging.info("加入负例,现在进度为"+str(i/self.new_test_length))
            # logging.info(illegal_edge_candi)
            self.test_edge.append(illegal_edge_candi)
            self.test_label.append(0)

    def get_cosine_distance_list(self):
        for edge in self.test_edge:
            node1 = edge[0]
            node2 = edge[1]
            node1_vec = np.array(self.vectors_dict.get(node1))
            node2_vec = np.array(self.vectors_dict.get(node2))
            # print(node1)
            # print(node2)
            cos = self.get_cosine_distance(node1_vec, node2_vec)
            self.test_cosine_distance.append(cos)

    def get_AUC(self):
        auc = roc_auc_score(self.test_label, self.test_cosine_distance)
        logging.debug("新旧测试集合法边比例为："+str(self.new_test_length/self.old_test_length))
        logging.debug("AUC分数为："+str(auc))

    @staticmethod
    def get_cosine_distance(node1_vec, node2_vec):
        num = float(np.sum(node1_vec * node2_vec))
        denom = np.linalg.norm(node1_vec) * np.linalg.norm(node2_vec)
        return num/denom

    def get_illegal_edge(self):

        node1 = self.node_list[np.random.choice(range(0, len(self.node_list)))]
        node2 = self.node_list[np.random.choice(range(0, len(self.node_list)))]
        curr_edge = [node1, node2]
        while(curr_edge in self.legal_edge or node1 not in self.vectors_dict or node2 not in self.vectors_dict):
            node1 = self.node_list[np.random.choice(range(0, len(self.node_list)))]
            node2 = self.node_list[np.random.choice(range(0, len(self.node_list)))]

            curr_edge = [node1, node2]
            # print(1)
        return curr_edge


if __name__ == '__main__':
    # enron:
    # python new_link_prediction.py ../output/withJUST/enron190_skip5/without_na_others/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5/without_na_others/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5/without_na_others/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1

    # python new_link_prediction.py ../output/withJUST/enron190_skip5_shuffle/without_na_others/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_shuffle/without_na_others/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_shuffle/without_na_others/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1

    # python new_link_prediction.py ../output/withJUST/enron190_skip5_init_in_window/without_na_others/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_init_in_window/without_na_others/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_init_in_window/without_na_others/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1

    # python new_link_prediction.py ../output/withJUST/enron190_skip5_no_init/without_na_others/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_no_init/without_na_others/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_no_init/without_na_others/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_fast_no_init/without_na_others/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_fast_no_init/without_na_others/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../output/withJUST/enron190_skip5_fast_no_init/without_na_others/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1

    # todo: dynamic_sg_gjw方法
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/enron190_skip5_full/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 2 1
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 2 1
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 2 1

    # todo: metapath2vec方法
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/enron_5_metapath/model/metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/enron_10_metapath/model/metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/enron_15_metapath/model/metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1

    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/enron_5metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/enron_10metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/enron_15metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/l40_enron_5metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/l40_enron_10metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/enron/l40_enron_15metapath_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1

    # 有问题: change2vec_gjw
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full/emb_merge/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full/emb_merge/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full/emb_merge/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 3 1

    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 3 1

    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb_merge/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb_merge/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb_merge/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 3 1

    # todo: online CTDNE方法
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2/0_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2/1_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2/2_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2_w5/0_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2_w5/1_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2_w5/2_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2_w1/0_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2_w1/1_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_full2_w1/2_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_new_w5/model/0_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_new_w5/model/1_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/enron_new_w5/model/2_model ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 1 1

    # todo: metapath方法
    # python new_link_prediction.py ../output/TKY_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/TKY_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/TKY_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1

    # todo: JUSTwithoutTRW方法
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/enron190_skip5_full/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/10.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/15.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/15.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/20.txt 2 1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/enron190_skip5_full/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/enron190_skip5_full/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/enron190_skip5_full/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/enron_5/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/enron_10/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/enron_15/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 2 1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/enron190_skip5_full/emb/5.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/10.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/enron190_skip5_full/emb/10.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/15.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/enron190_skip5_full/emb/15.emb ../dataset/slide_data_for_link_prediction/enron190_skip5_test_set/20.txt 0 1


    # TKY
    # python new_link_prediction.py ../output/withJUST/TKY190_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # python new_link_prediction.py ../output/withJUST/TKY190_full_p0/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_p0/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_p0/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_p0/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_p0/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # python new_link_prediction.py ../output/withJUST/TKY190_full_shuffle/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_shuffle/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_shuffle/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_shuffle/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_shuffle/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # python new_link_prediction.py ../output/withJUST/TKY190_full_init_in_window/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_init_in_window/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_init_in_window/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_init_in_window/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_init_in_window/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # python new_link_prediction.py ../output/withJUST/TKY190_full_no_init/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_no_init/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_no_init/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_no_init/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../output/withJUST/TKY190_full_no_init/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 2 1
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 2 1
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 2 1
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 2 1
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 2 1

    # python new_link_prediction.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 3 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 3 1

    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/metapath_100000model ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/metapath_200000model ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/metapath_300000model ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/metapath_400000model ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/metapath_500000model ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/l40_100000_metapath_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/l40_200000_metapath_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/l40_300000_metapath_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/l40_400000_metapath_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 1 1
    # python new_link_prediction.py ../baseline/output/metapath2vec/TKY/l40_500000_metapath_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 1 1

    # python new_link_prediction.py ../baseline/CTDNE/output/TKY_new_w10/model/9_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/TKY_new_w10/model/19_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/TKY_new_w10/model/29_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/TKY_new_w10/model/39_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 1 1
    # python new_link_prediction.py ../baseline/CTDNE/output/TKY_new_w10/model/49_model ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 1 1

    # python new_link_prediction.py ../output/TKY_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../output/TKY_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../output/TKY_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../output/TKY_full/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../output/TKY_full/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0  1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 2 1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190_full/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190_full/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/TKY_100000/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/TKY_200000/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/TKY_300000/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/TKY_400000/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/TKY_500000/emb/vec_init.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 2 1

    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/TKY190_full/emb/100000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/200000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/TKY190_full/emb/200000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/300000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/TKY190_full/emb/300000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/400000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/TKY190_full/emb/400000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/500000.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/modelwithStdSG/TKY190_full/emb/500000.emb ../dataset/slide_data_for_link_prediction/TKY190_test_set/555437.txt 0 1

    # 消融实验：
    # python new_link_prediction.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190/emb/vec_final.emb ../dataset/TKY/test_graph.txt 0 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190/emb/vec_final.emb ../dataset/TKY/test_graph.txt 2 1
    # python new_link_prediction.py ../baseline/ablation_experiment/output/JUST_static/TKY/emb/vec_init.emb ../dataset/TKY/test_graph.txt 2 1
    # baseline实验：
    # python new_link_prediction.py ../baseline/output/dynamic_sg_gjw/TKY190/emb/415368.emb ../dataset/TKY/test_graph.txt 2 1
    # python new_link_prediction.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/vec_final.emb ../dataset/TKY/test_graph.txt 3 1
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    isModel = int(sys.argv[3])  # 0表示自己生成的向量表示，1表示由模型函数save的模型表示，2表示标准sg模型save的向量表示，3代表自己生成的首位字母向量
    hasLabel = int(sys.argv[4])  # 0表示没有标签，1表示有标签

    lp = link_predictor()
    # logging.info(i)
    # lp.import_dataset("train_graph.txt", 0)  # 这里0和1是代表有没有标签
    # lp.import_model(str(i)+"_model", 1)  # 这次0123是代表用的哪个model
    # lp.import_model("vec_final.emb",0)
    # lp.import_model("temporal_model_skipgram", 1)
    lp.import_model(model_path, isModel)
    lp.get_test_set(test_path, hasLabel)  # 这里0和1是代表有没有标签
    # lp.import_model("../output/bib190/emb/vec_final.emb", 0)
    # lp.get_test_set("../dataset/bibsonomy-2ui/bibsonomy-2ui_test25.txt", 1)  # 这里0和1是代表有没有标签
    lp.get_cosine_distance_list()
    lp.get_AUC()