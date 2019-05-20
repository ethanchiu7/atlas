"""
this must be a map job
"""
import sys
sys.path.append('../conf')
sys.path.append('./')
import conf


def hash_dict(my_dict, hash_k):
    ret = {}
    for k in my_dict.keys():
        h = hash(k) % hash_k
        if h not in ret:
            ret[h] = {}
        ret[h][k] = my_dict[k]
    return ret


def parse_line(line, index_feature_dict):
    fea_lst = line.strip('\n').split('\t')
    if len(fea_lst) != FTR_NUM:
        return
    fea_value_dict = dict()
    for i in index_feature_dict:
        fea_value_dict[index_feature_dict[i]] = str(fea_lst[i])

    if len(fea_value_dict) <= 0:
        return
    return fea_value_dict


def get_language_one_hot(ori_fea_dict):
    assert 'lang' in ori_fea_dict
    language_list = ['english', 'hindi', 'xx', 'xx']
    fea_list = [1 if i == ori_fea_dict['lang'] else 0 for i in language_list]

    return fea_list


def get_app_one_hot(ori_fea_dict):
    assert 'app' in ori_fea_dict
    app_list = ['app_iflow', 'browser_iflow', 'app_video_immerse', 'browser_video_immerse']
    fea_list = [1 if i == ori_fea_dict['app'] else 0 for i in app_list]

    return fea_list


def get_chid_one_hot(ori_fea_dict):
    assert 'chid' in ori_fea_dict
    app_list = ['xx', 'xx', 'xx', 'xx']
    fea_list = [1 if i == ori_fea_dict['chid'] else 0 for i in app_list]

    return fea_list


def get_act_level_one_hot(ori_fea_dict):
    assert 'act_level' in ori_fea_dict
    act_level = int(ori_fea_dict['act_level'])
    fea_list = [1 if act_level == i else 0 for i in range(8)]
    return fea_list


def get_user_lt_top_cate_one_hot(ori_fea_dict):
    # assert 'user_lt_top_cate' in user_lt_top_cate
    top_cate = int(ori_fea_dict['top_c1'])
    fea_list = [1 if top_cate == i else 0 for i in range(100)]
    if sum(fea_list) == 0:
        fea_list[0] = 1
    return fea_list


def get_doc_fea(ori_fea_dict):
    fea_list = list()

    rt_pv = int(ori_fea_dict['rt_pv'])
    rt_clk = int(ori_fea_dict['rt_clk'])
    rt_ctr = float(ori_fea_dict['rt_ctr'])

    fea_list.append(rt_pv)
    fea_list.append(rt_clk)
    fea_list.append(rt_ctr)

    return fea_list


def get_relevant_fea(ori_fea_dict):
    fea_list = list()

    user_lt_top_cate = str(ori_fea_dict['top_c1'])

    cate_id_l1 = str(ori_fea_dict['c1'])
    cate_id_l2 = str(ori_fea_dict['c2'])

    rele_top_cate = 1 if (user_lt_top_cate in cate_id_l1 or user_lt_top_cate in cate_id_l2) else 0
    fea_list.append(rele_top_cate)

    return fea_list


def format_output_as_csv(uid, label, value_list):
    tmp_list = list()
    tmp_list.append(str(uid))
    tmp_list.append(str(label))
    tmp_list.extend(value_list)

    print('\t'.join(tmp_list))


def format_output_as_svm(uid, label, value_list):
    fea_kv_str_list = list()
    for i, j in enumerate(value_list):
        if float(j) != 0:
            fea_kv_str_list.append("{0}:{1}".format(i+1, j))

    if len(fea_kv_str_list) <= 0:
        return
    final_list = list()
    final_list.append(str(uid))
    final_list.append(str(label))
    final_list.extend(fea_kv_str_list)
    print(' '.join(final_list))


if __name__ == '__main__':

    FTR_NUM = len(conf.index_feature_dict)

    default_value = 'null'

    sample_dict = {}

    for line in sys.stdin:
        final_features_list = list()
        fea_value_dict = parse_line(line, conf.index_feature_dict)

        if fea_value_dict is None:
            continue
        assert 'label' in fea_value_dict
        label = str(fea_value_dict['label'])

        # --- user fea ---
        uid = str(fea_value_dict['uid'])

        # language one-hot
        final_features_list.extend(get_language_one_hot(fea_value_dict))
        # app one-hot
        final_features_list.extend(get_app_one_hot(fea_value_dict))
        # chid one-hot

        # lbs_prov one-hot
        # lbs_city one-hot
        # act_level one-hot
        final_features_list.extend(get_act_level_one_hot(fea_value_dict))
        # user_lt_top_cate
        final_features_list.extend(get_user_lt_top_cate_one_hot(fea_value_dict))

        # --- doc fea ---

        final_features_list.extend(get_doc_fea(fea_value_dict))

        # --- relevant fea ---
        final_features_list.extend(get_relevant_fea(fea_value_dict))

        # --- format output --
        final_features_list = [str(i) for i in final_features_list]
        format_output_as_csv(uid, label, final_features_list)
