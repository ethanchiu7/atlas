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


def dump_sample(sample_dict):

    # if len(sample_dict) != FTR_NUM:
    #     return

    sample_feature_list = [sample_dict.get(conf.index_feature_dict[i], conf.DEFAULT_VALUE) for i in range(FTR_NUM)]
    print('\t'.join(sample_feature_list))

    return


if __name__ == '__main__':

    FTR_NUM = len(conf.feature_index_dict)

    default_value = 'null'

    sample_dict = {}

    for line in sys.stdin:
        line = line.strip()
        if '[ditt]' in line:
            if len(sample_dict) > 0:
                dump_sample(sample_dict)
                sample_dict = {}
        elif '=0:' in line:
            kv_list = line.split('=0:', 1)
            if len(kv_list) != 2:
                continue
            k, v = kv_list
            if k in conf.feature_index_dict:
                sample_dict[k] = v
        else:
            # print(line)
            continue

    if len(sample_dict) > 0:
        dump_sample(sample_dict)




