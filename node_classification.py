#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @file   : node_prediction.py
# @author : Zhutian Lin
# @date   : 2019/10/17
# @version: 1.0
# @desc   : 完成点分类任�?
import pandas as pd
import numpy as np
import logging
import sklearn
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # 日志格式化输�?
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"                        # 日期格式
fp = logging.FileHandler('config.log', encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fs, fp])    # 调用handlers=[fp,fs]
#  输出到命令行�?
#  计时�?
import time
import sys


class node_classilier:
    def __init__(self):
        logging.info("初始化分类器")
        self.vectors_dict = {}  # 每个点对应的字典{点index(str):向量(nparray)} 这个str可不能带小数�?
        self.node_dict = {}  # 每个点的label
        self.node_list = []  # 备选点
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
    # 操作：首先导入数据集，格式不变；原来的test_set输入为以空格为分隔符的node1 node2，主程序的embedding，格式不�?

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
            self.node_list = word_vectors.wv.vocab.keys()
            for node_id in self.node_list:
                tmp = {node_id: word_vectors[node_id]}
                # tmp = {node_id: list(word_vectors[node_id])}
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

    def import_node(self, dsname):
        # 统一操作为str
        # node_data = np.loadtxt(dsname, delimiter="\t").tolist()
        # node_data = np.loadtxt(dsname, delimiter=",").tolist()
        node_data = np.loadtxt(dsname, dtype=str, delimiter="\t")
        # node_data = np.loadtxt(dsname, dtype=str, delimiter="\t").tolist()
        # node_data = np.loadtxt(dsname, dtype=str, delimiter=",").tolist()
        for edge in node_data:
            tmp_1 = {edge[0]: edge[1]}
            tmp_2 = {edge[2]: edge[3]}
            # tmp_1 = {str(int(edge[0])): str(int(edge[1]))}
            # tmp_2 = {str(int(edge[2])): str(int(edge[3]))}
            self.node_dict.update(tmp_1)
            self.node_dict.update(tmp_2)

        node_data = []

    def build_train_test(self):
        logging.debug("build_train_test")
        vec = []
        label = []
        for node in self.vectors_dict:
            node_vec = self.vectors_dict.get(node)
            if type(node_vec) != list:
                node_vec = node_vec.tolist()
            # node_vec = node_vec.tolist()
            if node in self.vectors_dict and node in self.node_dict:  # 去掉没有压中
                vec.append(node_vec)
                label.append(self.node_dict.get(node))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(vec, label, test_size=0.25)

    def classify(self):
        logging.debug("classify")
        w = []
        a = []
        i = []
        

        lr = LogisticRegression(C=1000.0, random_state=0)
        lr.fit(self.X_train, self.y_train)
        Y_predict_lr = lr.predict(self.X_test).tolist()
        mif1_lr = f1_score(self.y_test, Y_predict_lr, average='micro')
        maf1_lr = f1_score(self.y_test, Y_predict_lr, average='macro')
        f1_lr = f1_score(self.y_test, Y_predict_lr, average='weighted')
        
        # print("经过lr训练的预测情况为�?+str(f1_lr))
        print("weighted  "+str(f1_lr))
        print("micro  "+str(mif1_lr))
        print("macro  "+str(maf1_lr))

        w.append(str(f1_lr))
        a.append(str(maf1_lr))
        i.append(str(mif1_lr))
          
        with open('result.txt','a') as f:
            f.write('/'.join(w)+':weighted\n')
            f.write('/'.join(a)+':macro\n')
            f.write('/'.join(i)+':micro\n')
        # svm = SVC(kernel='linear', C=1.0, random_state=0)
        # svm.fit(self.X_train, self.y_train)
        # Y_predict_svm = svm.predict(self.X_test).tolist()
        # f1_svm = f1_score(self.y_test,Y_predict_svm,average='weighted')
        # print("经过svm训练的预测情况为�?+str(f1_svm))


if __name__ == '__main__':
    # experiment
    # python node_classification.py ../output/withJUST/TKY180/emb/vec_final.emb ../dataset/TKY/train_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190/emb/vec_final.emb ../dataset/TKY/train_graph.txt 0

    # python node_classification.py ../output/withJUST/TKY190_full/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190_full_shuffle/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190_full_init_in_window/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190_full_no_init/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190_full_p1/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190_full_p5/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../output/withJUST/TKY190_full_p0/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0

    # 消融实验�?
    # python node_classification.py ../baseline/ablation_experiment/output/TRWwithoutJUST/TKY190_full/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # python node_classification.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/TKY190_full/emb/vec_final.emb ../dataset/TKY/full_graph.txt 2
    # python node_classification.py ../baseline/ablation_experiment/output/JUST_static/TKY_555437/emb/vec_init.emb ../dataset/TKY/full_graph.txt 2
    # python node_classification.py ../baseline/ablation_experiment/output/modelwithStdSG/TKY190_full/emb/vec_final.emb ../dataset/TKY/full_graph.txt 0
    # baseline实验�?
    # python node_classification.py ../baseline/output/dynamic_sg_gjw/TKY190_full/emb/555436.emb ../dataset/TKY/full_graph.txt 2
    # python node_classification.py ../baseline/output/change2vec_gjw/TKY190_full/emb_merge/vec_final.emb ../dataset/TKY/full_graph.txt 3
    # python node_classification.py ../baseline/CTDNE/output/TKY_new_w10/model/54_model ../dataset/TKY/full_graph.txt 1
    # python node_classification.py ../baseline/output/metapath2vec/TKY/metapath_wholemodel ../dataset/TKY/full_graph.txt 1
    # python node_classification.py ../baseline/output/metapath2vec/TKY/l40_whole_metapath_model ../dataset/TKY/full_graph.txt 1

    # enron:
    # python node_classification.py ../output/withJUST/enron190_skip5/without_na_others/emb/vec_final.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # python node_classification.py ../output/withJUST/enron190_skip5_shuffle/without_na_others/emb/vec_final.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # python node_classification.py ../output/withJUST/enron190_skip5_init_in_window/without_na_others/emb/vec_final.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # python node_classification.py ../output/withJUST/enron190_skip5_no_init/without_na_others/emb/vec_final.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # python node_classification.py ../output/withJUST/enron190_skip5_fast_no_init/without_na_others/emb/20.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # 消融实验�?
    # python node_classification.py ../baseline/ablation_experiment/output/TRWwithoutJUST/enron190_skip5_full/emb/vec_final.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # todo:python node_classification.py ../baseline/ablation_experiment/output/JUSTwithoutTRW/enron190_skip5_full/emb/???.emb ../dataset/sig_enron/enron_dele_na_others.txt 2
    # python node_classification.py ../baseline/ablation_experiment/output/JUST_static/enron/emb/vec_init.emb ../dataset/sig_enron/enron_dele_na_others.txt 2
    # python node_classification.py ../baseline/ablation_experiment/output/modelwithStdSG/enron190_skip5_full/emb/vec_final.emb ../dataset/sig_enron/enron_dele_na_others.txt 0
    # baseline实验�?
    # todo:python node_classification.py ../baseline/output/dynamic_sg_gjw/enron190_skip5_full/emb/???.emb ../dataset/sig_enron/enron_dele_na_others.txt 2
    # fixme:python node_classification.py ../baseline/output/change2vec_gjw/enron190_skip5_full/emb_merge/20.emb ../dataset/sig_enron/enron_dele_na_others.txt 3
    # python node_classification.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb/20.emb ../dataset/sig_enron/enron_dele_na_others.txt 3
    # python node_classification.py ../baseline/output/change2vec_gjw/enron190_skip5_full_has0/emb_merge/20.emb ../dataset/sig_enron/enron_dele_na_others.txt 3
    # python node_classification.py ../baseline/output/metapath2vec/enron/enron_20_metapath/model/metapath_model ../dataset/sig_enron/enron_dele_na_others.txt 1
    # python node_classification.py ../baseline/output/metapath2vec/enron/enron_20metapath_model ../dataset/sig_enron/enron_dele_na_others.txt 1
    # python node_classification.py ../baseline/output/metapath2vec/enron/l40_enron_20metapath_model ../dataset/sig_enron/enron_dele_na_others.txt 1
    # python node_classification.py ../baseline/output/metapath2vec/enron/metapath_model_other ../dataset/sig_enron/enron_dele_na_others.txt 1

    # python node_classification.py ../baseline/CTDNE/output/enron_full2_w5/3_model ../dataset/sig_enron/enron_dele_na_others.txt 1
    # python node_classification.py ../baseline/CTDNE/output/enron_full2/3_model ../dataset/sig_enron/enron_dele_na_others.txt 1
    # python node_classification.py ../baseline/CTDNE/output/enron_full2_w1/3_model ../dataset/sig_enron/enron_dele_na_others.txt 1
    # python node_classification.py ../baseline/CTDNE/output/enron_new_w5/model/3_model ../dataset/sig_enron/enron_dele_na_others.txt 1


    model_path = sys.argv[1]
    node_path = sys.argv[2]
    isModel = int(sys.argv[3])  # 0表示自己生成的向量表示，1表示由模型函数save的模型表示，2表示标准sg模型save的向量表示，3代表自己生成的首位字母向�?

    nlf = node_classilier()
    nlf.import_model(model_path, isModel)
    nlf.import_node(node_path)
    # nlf.import_model("29_model", 1)
    # nlf.import_node("train_graph.txt")
    # nlf.import_model("../output/bib190/emb/vec_final.emb", 0)
    # nlf.import_node("../dataset/bibsonomy-2ui/bibsonomy-2ui_75percent_tag2int.txt")
    # nlf.import_model("../output/withJUST/TKY180/emb/vec_final.emb", 0)
    # nlf.import_node("../dataset/TKY/train_graph.txt")
    nlf.build_train_test()
    nlf.classify()
