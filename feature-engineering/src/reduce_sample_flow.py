import sys
sys.path.append('../conf')
sys.path.append('./')
from utils import MapReduceBase
import conf


def split_time_win(action_list):
    if not isinstance(action_list, list):
        return
    if not len(action_list) > 0:
        return
    action_field_list = list()

    last_act_field = list()
    last_tm = 0
    for act in action_list:
        curr_tm = int(act['tm'])
        if curr_tm <= 0:
            # continue
            curr_tm = 0
        if last_tm == 0:
            last_act_field.append(act)
            last_tm = curr_tm
            continue
        if curr_tm > last_tm + TIME_WIN:
            # clear
            # if sum([int(act['read']) for act in last_act_field]) >= 1:
            #     action_field_list.append(last_act_field)
            last_act_field = list()
        last_act_field.append(act)
        last_tm = curr_tm
    if len(last_act_field) > 0:
        action_field_list.append(last_act_field)

    return action_field_list


def print_sample(samples):
    for sample in samples:
        value_dict = dict()
        for i in conf.index_feature_dict:
            value_dict[i] = sample.get(conf.index_feature_dict[i], conf.DEFAULT_VALUE)
        sorted_feature = [str(i[1]) for i in sorted(value_dict.items(), key=lambda x: int(x[0]))]

        print('\t'.join(sorted_feature))


def generate_sample(action_list):
    return action_list
    # this item which is below the positive is negative
    single_user_samples = list()
    positive_num = 0
    negative_num = 0
    # time reverse
    # action_list.reverse()

    # positive
    for act in action_list:
        if str(act['read']) == '1':
            single_user_samples.append(act)
            positive_num += 1

    if len(single_user_samples) <= 0:
        return single_user_samples

    # negative
    for act in action_list:
        # if negative_num >= (2*positive_num):
        #     break
        if str(act['read']) == '0':
            single_user_samples.append(act)
            negative_num += 1

    return single_user_samples


class SampleMapReduce(MapReduceBase):

    def map_process(self):
        print(len(self.current_value_dict))

    def reduce_process(self):
        if len(self.value_dict_list) <= 0:
            return
        # sort
        # user_action_list = sorted(self.value_dict_list, key=lambda x: x['tm'], reverse=True)[:ACTION_LIMIT].reverse()
        user_action_list = sorted(self.value_dict_list, key=lambda x: x['tm'], reverse=True)[:ACTION_LIMIT]
        # reverse
        user_action_list.reverse()
        # split by time window
        action_field_list = split_time_win(user_action_list)
        if action_field_list is None:
            return

        if len(action_field_list) <= 0:
            return

        for i in action_field_list:
            sample_list = generate_sample(i)
            if sample_list is None or len(sample_list) <= 0:
                continue
            print_sample(sample_list)


if __name__ == '__main__':

    ACTION_LIMIT = 10000

    TIME_WIN = 10*60

    reducer = SampleMapReduce(role_type='reducer', seq='\t', key_index=0, index_name_dict=conf.index_feature_dict)

    reducer(sys.stdin)
    # reducer(open('test.in'))
