import optparse
import util
import numpy as np
from sklearn.metrics import roc_auc_score


def get_cosine_distance(node1_vec, node2_vec):
    num = np.float(np.sum(node1_vec * node2_vec))
    denom = np.linalg.norm(node1_vec) * np.linalg.norm(node2_vec)
    return num / denom


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-s', dest='s', help='Test-set', type='str', default='test.txt')
    parser.add_option('-o', dest='o', help='Neg-set', type='str', default='/result')
    parser.add_option('-m', dest='m', help='model', type='str', default='/emb/model')
    options, args = parser.parse_args()

    id2vec = util.import_model(options.m)
    pos_edges, nd1 = util.import_edges(options.s, '5')
    neg_edges, nd2 = util.import_edges(options.o, '3')
    st_vec = []
    sec_vec = []
    label = []
    print("处理正例")
    pos_cnt = 0
    neg_cnt = 0
    for edge in pos_edges:
        if edge[0] in id2vec and edge[1] in id2vec:
            st_vec.append(id2vec.get(edge[0]))
            sec_vec.append(id2vec.get(edge[1]))
            label.append(1)
            pos_cnt += 1
    print("处理负例")
    for edge in neg_edges:
        if edge[0] in id2vec and edge[1] in id2vec:
            st_vec.append(id2vec.get(edge[0]))
            sec_vec.append(id2vec.get(edge[1]))
            label.append(0)
            neg_cnt += 1

    st_vec = np.array(st_vec)
    sec_vec = np.array(sec_vec)

    cos_arr = []
    for i in range(len(st_vec)):
        cos_arr.append(get_cosine_distance(st_vec[i], sec_vec[i]))
    print("正例命中率：",
          pos_cnt / len(pos_edges))
    print("负例命中率：",
          neg_cnt / len(neg_edges))
    print("总命中率：",
          len(cos_arr) / (len(pos_edges) + len(neg_edges)) if len(cos_arr) <=
                                                              len(pos_edges) + len(neg_edges) else 1)
    if neg_cnt != 0:
        print("正负例真实比例:", pos_cnt / neg_cnt)
    else:
        print("正负例真实比例: 负例数为0 . ")

    print("AUC", roc_auc_score(label, cos_arr))
# -o experiment/emb/link_prediction_folder/tky/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/200000.txt -m experiment/emb/link_prediction_folder/tky/online_model/online_9_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/300000.txt -m experiment/emb/link_prediction_folder/tky/online_model/online_19_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/400000.txt -m experiment/emb/link_prediction_folder/tky/online_model/online_29_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/4_5_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/500000.txt -m experiment/emb/link_prediction_folder/tky/online_model/online_39_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/5_end_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/555438.txt -m experiment/emb/link_prediction_folder/tky/online_model/online_49_model

# -o experiment/emb/link_prediction_folder/tky/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/200000.txt -m experiment/emb/link_prediction_folder/tky/metapath_model/100000_metapath_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/300000.txt -m experiment/emb/link_prediction_folder/tky/metapath_model/200000_metapath_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/400000.txt -m experiment/emb/link_prediction_folder/tky/metapath_model/300000_metapath_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/4_5_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/500000.txt -m experiment/emb/link_prediction_folder/tky/metapath_model/400000_metapath_model
# -o experiment/emb/link_prediction_folder/tky/neg_txt/5_end_neg.txt -s experiment/emb/link_prediction_folder/tky/window_txt/555438.txt -m experiment/emb/link_prediction_folder/tky/metapath_model/500000_metapath_model


# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta01/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta01/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta01/emb/15.emb

# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta1/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta1/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta1/emb/15.emb


# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta3/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta3/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta3/emb/15.emb

# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta5/emb/5.emb'
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta5/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta5/emb/15.emb

# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta7/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta7/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta7/emb/15.emb

# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta9/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta9/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta9/emb/15.emb

# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta99/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/2_3_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta99/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/enron190_skip5/beta99/emb/15.emb


# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/metapath2vec_enron_series/model/5_metapath_model
# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/metapath2vec_enron_series/model/10_metapath_model
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/metapath2vec_enron_series/model/15_metapath_model


# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/change2vec_enron_has0/emb/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/change2vec_enron_has0/emb/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/change2vec_enron_has0/emb/15.emb


# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/10.txt -m experiment/emb/link_prediction_folder/enron/model/change2vec_enron_has0/emb_merge/5.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/1_2_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/15.txt -m experiment/emb/link_prediction_folder/enron/model/change2vec_enron_has0/emb_merge/10.emb
# -o experiment/emb/link_prediction_folder/enron/neg_txt/3_4_neg.txt -s experiment/emb/link_prediction_folder/enron/window_txt/20.txt -m experiment/emb/link_prediction_folder/enron/model/change2vec_enron_has0/emb_merge/15.emb
