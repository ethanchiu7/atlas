#coding: utf8
import sys
import os
import json
import time


def general_format(db_key, value_list):
    """
    eg: index_name=kExp_9999`index_data=111111,7.0#2222222,2.3#333333,1.3
    :param db_key: which can be index for DB
    :param value_list: list of (item, score) tuple
    :return:
    """
    tmp = "index_name={}`index_data={}"

    item_score_list = list()
    for i in value_list:
        item_id, item_score = "", "1.0"
        if isinstance(i, (tuple, list)) and len(i) == 2:
            item_id, item_score = str(int(i[0])), str(float(i[1]))
        elif isinstance(i, dict) and 'item_id' in i:
            item_id = str(int(i['item_id']))
        elif isinstance(i, (int, str, float)) and len(str(i)) > 0:
            item_id = str(int(i))
            item_score = "1.0"
        else:
            continue
        item_score_list.append((item_id, item_score))

    if len(item_score_list) == 0:
        return
    kv_str_list = ["{},{}".format(i[0], i[1]) for i in item_score_list]
    index_data = '#'.join(kv_str_list)

    return tmp.format(db_key, index_data)


def write_to_file(write_file, input_line):
    if write_file is None or input_line is None:
        return
    if isinstance(input_line, str):
        line = input_line
    elif isinstance(input_line, (tuple, list)):
        line = '\t'.join(input_line)
    else:
        return
    with open(write_file, 'a') as f:
        f.write(line)
        f.write('\n')


def formate4pangu(db_key, value_list):
    """
    eg: index_name=kExp_9999`index_data=111111,7.0#2222222,2.3#333333,1.3
    :param db_key: which can be index for DB
    :param value_list: list of (item, score) tuple
    :return:
    """
    tmp = "index_name={}`index_data={}"
    kv_str_list = ["{},{}".format(str(int(i[0])), str(float(i[1]))) for i in value_list]
    index_data = '#'.join(kv_str_list)

    return tmp.format(db_key, index_data)


def formate_kSmTrending4pangu(db_key, value_list):
    """
    eg: index_name=kExp_9999`index_data=111111,7.0#2222222,2.3#333333,1.3
    :param db_key: which can be index for DB
    :param value_list:
    :return:
    """
    tmp = "index_name={}`index_data={}"
    kv_str_list = ["{},{}".format(i['item_id'], 1) for i in value_list]
    index_data = '#'.join(kv_str_list)

    return tmp.format(db_key, index_data)


class ChainInterceptor(object):
    """
    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kCtrlItem", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='datamining')
    _interceptor(chain_key=None, item_list=None)
    """

    def __init__(self, chain_type="kCtrlItem", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=False, exe_user='iflow'):
        # 写入文件的行数限制，大于则触发更新pangu
        self.__write_file_buffer = 7777
        self._chain_type = chain_type
        self._translator_project = translator_project

        self._write_file = str(translator_project) + "/data/" + chain_type + ".multilang." + str(int(time.time()))
        self._log_file = str(translator_project) + "/log/remove_" + chain_type + "." + str(int(time.time())) + ".log"
        self._update_shell = str(translator_project) + "/bin/intercepted-remover.sh"
        self._update_pangu = update_pangu
        self._exe_user = exe_user

        self._write_count = 0

        print("ChainInterceptor going to write file {}".format(self._write_file))
        print("ChainInterceptor log file {}".format(self._log_file))

        self.write_folder = os.path.dirname(self._write_file)
        if not os.path.exists(self.write_folder):
            print("mkdir -p {}".format(self.write_folder))
            os.system("mkdir -p {}".format(self.write_folder))
        self.log_folder = os.path.dirname(self._log_file)
        if not os.path.exists(self.log_folder):
            print("mkdir -p {}".format(self.log_folder))
            os.system("mkdir -p {}".format(self.log_folder))

        if os.path.exists(self._write_file):
            #  remove to pangu
            if self._update_pangu and self._update_shell is not None and os.path.exists(self._update_shell):
                self.update()
                # print("sh {update_shell} {exe_user} {chain_type} {write_file} > {log_file} 2>&1"
                #       .format(update_shell=self._update_shell, exe_user=self._exe_user, chain_type=self._chain_type,
                #               log_file=self._log_file, write_file=self._write_file))
                # os.system("sh {update_shell} {exe_user} {chain_type} {write_file} > {log_file} 2>&1"
                #           .format(update_shell=self._update_shell, exe_user=self._exe_user, chain_type=self._chain_type,
                #                   log_file=self._log_file, write_file=self._write_file))
                # print("move to pangu finish !")
                # print("remove file: {}".format(self._write_file))
                # if os.path.exists(self._write_file):
                #     os.remove(self._write_file)
        else:
            print("write_file not exist: {}".format(self._write_file))

    def update(self):
        if os.path.exists(self._write_file):
            #  update to pangu
            if self._update_pangu and self._update_shell is not None and os.path.exists(self._update_shell):
                print("update pangu while write_file_buffer : {}, curr write_count : {}".format(self.__write_file_buffer, self._write_count))
                print("Trigger Update To Pangu [synchronous] !!!")
                result_file = os.path.split(self._write_file)[1]
                print("sh {update_shell} {exe_user} {chain_type} {result_file} > {log_file} 2>&1"
                      .format(update_shell=self._update_shell, exe_user=self._exe_user, chain_type=self._chain_type,
                              result_file=result_file, log_file=self._log_file))
                os.system("sh {update_shell} {exe_user} {chain_type} {result_file} > {log_file} 2>&1"
                          .format(update_shell=self._update_shell, exe_user=self._exe_user, chain_type=self._chain_type,
                                  result_file=result_file, log_file=self._log_file))
                print("update to pangu finish !")
                print("remove file: {}".format(self._write_file))

                # init write_file
                self._write_file = str(self._translator_project) + "/data/" + self._chain_type + "." + str(int(time.time()))
                self._log_file = str(self._translator_project) + "/log/remove_" + self._chain_type + "." + str(
                    int(time.time())) + ".log"
                self._write_count = 0
                print("init write_file : {}".format(self._write_file))

                # if os.path.exists(self._write_file):
                #     os.remove(self._write_file)
                #     self._write_count = 0

    def __call__(self, chain_key=None, item_list=None):
        if chain_key is None or item_list is None:
            print("chain_key or item_list is None")
            return
        if not isinstance(item_list, list):
            print("item_list type is not list")
            return
        if len(item_list) < 1:
            print("item_list len < 1")
            return
        line = general_format(db_key=chain_key, value_list=item_list)
        write_to_file(write_file=self._write_file, input_line=line)
        self._write_count += 1
        if self._write_count % 10 == 1:
            print("ChainInterceptor write_count:{a}, curr_line:{b}".format(a=self._write_count, b=line))
        if self._write_count >= self.__write_file_buffer:
            self.update()

    def __del__(self):
        if os.path.exists(self._write_file):
            #  remove to pangu
            if self._update_pangu and self._update_shell is not None and os.path.exists(self._update_shell):
                self.update()
                # print("sh {update_shell} {exe_user} {chain_type} > {log_file} 2>&1"
                #       .format(update_shell=self._update_shell, exe_user=self._exe_user, chain_type=self._chain_type, log_file=self._log_file))
                # os.system("sh {update_shell} {exe_user} {chain_type} > {log_file} 2>&1"
                #           .format(update_shell=self._update_shell, exe_user=self._exe_user, chain_type=self._chain_type, log_file=self._log_file))
                # print("move to pangu finish !")
                # print("remove file: {}".format(self._write_file))
                # if os.path.exists(self._write_file):
                #     os.remove(self._write_file)
        else:
            print("write_file not exist: {}".format(self._write_file))


class Interceptor(object):
    """
    from interceptor import Interceptor
    interceptor = Interceptor(folder='/home/datamining/tuixing.zx/remover4pangu/data', file_name='kExplore.txt', update_pangu=True, update_shell='')
    for i in range(10):
        interceptor.write_kExplore_to_remover(input_obj=("key__xx", "value...xx"))
    """

    def __init__(self, file_path='./intercept.txt', seq='\t', folder=None, file_name=None, update_pangu=False, update_shell=None, chain_name=None):

        self._file_path = None
        self._seq = '\t'
        self._chain_name = None
        self._update_pangu = None
        self._update_shell = None

        self._write_count = 0

        if file_path is not None:
            self._file_path = file_path
        if seq is not None:
            self._seq = seq
        if chain_name is not None:
            self._chain_name = chain_name
        if update_pangu is not None:
            self._update_pangu = update_pangu
        if update_shell is not None:
            self._update_shell = update_shell

        if folder is not None and file_name is not None:
            self._file_path = os.path.join(folder, file_name)

        self.folder = os.path.dirname(self._file_path)

        if not os.path.exists(self.folder):
            print("mkdir -p {}".format(self.folder))
            os.system("mkdir -p {}".format(self.folder))

        if os.path.exists(self._file_path):
            #  remove to pangu
            if self._update_pangu and self._update_shell is not None:
                shell_path, shell_name = os.path.split(self._update_shell)
                log_path = os.path.join(os.path.dirname(shell_path), 'log')
                log_file = os.path.join(log_path, shell_name.split('.')[0])
                print("sh {update_shell} > {log_file}.log".format(update_shell=self._update_shell, log_file=log_file))
                os.system("sh {update_shell} > {log_file}.log".format(update_shell=self._update_shell, log_file=log_file))
                print("remove to pangu finish !")
            print("remove file: {}".format(self._file_path))
            os.remove(self._file_path)
        else:
            print("file not exists: {}".format(self._file_path))
        print("tuixing.zx : The Interceptor will print data to : {}".format(self._file_path))

    def general_write(self, chain_key=None, item_list=None):
        if chain_key is None or item_list is None:
            return
        if not isinstance(item_list, list):
            return
        if len(item_list) < 1:
            return
        line = general_format(db_key=chain_key, value_list=item_list)
        self.write_to_file(line)

    def write_kExplore_to_remover(self, chain_key=None, item_list=None):
        if chain_key is None or item_list is None:
            return
        line = formate4pangu(db_key=chain_key, value_list=item_list)
        self.write_to_file(line)

    def write_kSmTrending_to_remover(self, chain_key=None, item_list=None):
        if chain_key is None or item_list is None:
            return
        line = formate_kSmTrending4pangu(db_key=chain_key, value_list=item_list)
        self.write_to_file(line)

    def write_to_file(self, input_line):
        line = ""
        if isinstance(input_line, str):
            line = input_line
        elif isinstance(input_line, (tuple, list)):
            line = self._seq.join(input_line)
        else:
            return
        with open(self._file_path, 'a') as f:
            f.write(line)
            f.write('\n')
            self._write_count += 1

    def __call__(self, input_obj):

        self.write_to_file(input_obj)


def test():
    """

    todo 2019-04-01 18:00 kExplorekeyword / kExploreTag / kExploreTopic / kExploreL2Ctg / kExploreL1Ctg - success
    s68: /home/iflow/gz_team1/product/explore_chain_wuyue/py
        py/update_wuyue_explore_chains.py

    todo 2019-04-01 20:41 kSmTrendingTop / kSmTrendingTag / kSmTrendingCat - success
    s68: /home/iflow/bj_team1/wangkun
        sm_trend/sm_cluster_chain.py
    add row337 :

    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kSmTrending", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='iflow')
    _interceptor(chain_key=None, item_list=None)

    todo 2019-04-03 17:40 kCategoryRankT1 - success
    s68 iflow /home/iflow/gz_team1/product/offline-recommend-ugc/category_rank2/bin/python/predict.py

    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kCategoryRankT1", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='iflow')
    _interceptor(chain_key=None, item_list=None)

    todo 2019-04-03 17:40 kCatgoryRankT2 - success
    s68 iflow /home/iflow/gz_team1/product/offline-recommend-ugc/category_rank2/bin/python/t2_predict.py

    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kCatgoryRankT2", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='iflow')
    _interceptor(chain_key=None, item_list=None)

    todo 2019-04-01 20:53 kRdtmTop / kRdtmCat / kRfCat - success
    s68: /home/iflow/zhengzl/app_recall
        chain.py
    add row 147:
    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kRdtmCat", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='iflow')
    _interceptor(chain_key=None, item_list=None)

    todo 2019-04-03 10:53 kHighQuality - success
    s67 /home/iflow/gz_team1/product/offline-recommend/rec_high_quality_stream/src/share_rec_pusher.py
    add row 116:
    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kHighQuality", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='iflow')
    _interceptor(chain_key=None, item_list=None)

    todo 2019-04-03 17:40 kGeneralRank - success
    s68 iflow
    from interceptor import ChainInterceptor
    _interceptor = ChainInterceptor(chain_type="kGeneralRank", translator_project="/home/iflow/tuixing.zx/remover4pangu", update_pangu=True, exe_user='iflow')
    _interceptor(chain_key=None, item_list=None)


    :return:
    """
    # interceptor = Interceptor(file_path='./intercept.txt', seq='\t')
    # interceptor = Interceptor(folder='data', file_name='kExplore.txt')
    # for i in range(10):
    #     interceptor.write_kExplore_to_remover(chain_key="key__xx", item_list=[[123, 0.8], [456, 0.6]])
    #     # interceptor(input_obj=("key__xx", "value...xx"))

    interceptor = Interceptor(folder='data', file_name='kSmTrending.txt')
    for i in range(10):

        interceptor.write_kSmTrending_to_remover(chain_key="key__xx", item_list=[{'item_id': 123}, {'item_id': 678}])
        # interceptor(input_obj=("key__xx", "value...xx"))


if __name__ == '__main__':

    test()
