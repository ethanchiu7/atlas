# -*- coding: utf-8 -*-
"""
Author by tuixing.zx
load file tools
This utils just for python3
"""
import os
import sys
import json
# for python2 MapReduce job, this is needed
import argparse
sys.path.append('./')


class MapReduceBase(object):

    def __init__(self, role_type='mapper', seq='\t', key_index=0, columns_num=None, is_json=False, index_name_dict=None, **kwargs):
        """
        :param role_type:    mapper / reducer / map / reduce
        :param seq:
        :param key_index:
        :param columns_num:
        :param kwargs:  index_name_dict

        e.g.

        class TestMapReduce(MapReduceBase):

            def __init__(self, **kwargs):
                super(TestMapReduce, self).__init__(**kwargs)
                self.translator = kwargs.get('translator')
            def map_process(self):
                print(len(self.current_value_dict))
            def reduce_process(self):
                print(len(self.value_dict_list))

        index_name_dict = {
                            1: 'name1',
                            2: 'name2',
                            7: 'name7',
                            }

        mr = MapReduceBase(role_type='mapper', index_name_dict=index_name_dict)
        rm(sys.stdin)

        """
        # private attribute
        self.__type = role_type
        self.__seq = seq
        self.__key_index = key_index
        self.__columns_num = 0
        if columns_num is not None and columns_num >= 1:
            self.__columns_num = columns_num

        self.__is_json = is_json

        self.__index_name_dict = None

        # These properties should be set by the user via keyword arguments.
        self.current_key = None
        self.current_value_dict = dict()

        self.previous_key = None
        self.value_dict_list = list()
        """
        allowed_kwargs = {
                          'index_name_dict',
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        """

        if index_name_dict is not None and isinstance(index_name_dict, dict):
            self.__index_name_dict = index_name_dict
        else:
            assert columns_num > 0
            self.__index_name_dict = dict([(i, str(i)) for i in range(columns_num)])

        if self.__columns_num <= 0 and len(self.__index_name_dict) > 0:
            self.__columns_num = len(self.__index_name_dict)

        assert len(self.__index_name_dict) > 0

    def map_process(self):
        """
        This should map the self.current_value_dict
        :return:
        """

        print(self.current_value_dict[0])

    def reduce_process(self):
        """
        This should reduce the self.value_dict_list
        :return:
        """

        print(len(self.value_dict_list))

    def __call__(self, input_io):
        """
        :param input_io: <class '_io.TextIOWrapper'>
        :return:
        """
        for input_line in input_io:

            if self.__is_json and self.__type in ['mapper', 'map']:
                self.current_key = eval(input_line)
                self.current_value_dict = eval(input_line)
                self.map_process()
                continue

            # process line as string
            if self.__index_name_dict is None or len(self.__index_name_dict) == 0:
                return

            if not isinstance(input_line, str):
                continue
            columns = input_line.split(self.__seq)
            if len(columns) != self.__columns_num:
                continue

            self.current_key = columns[self.__key_index].strip()

            # generate std value dict for each line
            for i in self.__index_name_dict:
                self.current_value_dict[self.__index_name_dict[i]] = columns[i]

            if self.__type in ['mapper', 'map']:
                self.map_process()
                continue
            elif self.__type in ['reducer', 'reduce']:
                if self.current_key == self.previous_key:
                    pass
                else:
                    if len(self.value_dict_list) > 0:
                        self.reduce_process()
                        self.value_dict_list = list()
                self.value_dict_list.append(self.current_value_dict)
                self.previous_key = self.current_key
                continue
            else:
                return
        if self.__type in ['reducer', 'reduce'] and len(self.value_dict_list) > 0:
            self.reduce_process()
            self.value_dict_list = list()


def load_columns_to_dict(file_path, seq='\t', key_index=0, value_index=1, columns_num=2):
    result_dict = dict()
    for line in open(file_path, 'r'):
        columns = line.split(seq)
        if len(columns) != columns_num:
            continue
        key = columns[key_index].strip()
        value = columns[value_index].strip()
        result_dict[key] = value
    if len(result_dict) <= 0:
        print("load_columns_to_dict failed , file_path: {}".format(file_path))
        sys.exit(1)
    return result_dict


def load_columns_to_nested_dict(file_path, seq='\t', key_index=0, columns_num=2, **kwargs):
    """

    :param file_path:
    :param seq:
    :param key_index:
    :param columns_num:
    :param kwargs:
    :return:
    """
    result_dict = dict()
    for line in open(file_path, 'r'):
        columns = line.split(seq)
        if len(columns) != columns_num:
            continue
        key = columns[key_index].strip()
        if 'index_name_dict' in kwargs:
            index_name_dict = kwargs['index_name_dict']
        else:
            index_name_dict = dict([(i, str(i)) for i in range(columns_num)])

        # generate std value dict for each line
        line_value_dict = dict()
        for i in index_name_dict:
            line_value_dict[index_name_dict[i]] = columns[i]
        result_dict[key] = line_value_dict

    return result_dict


def json2dict(json_str, encoding='utf8'):
    data = None
    try:
        data = json.loads(json_str, encoding=encoding)
    except Exception as ex:
        # works in python3
        # print("json2dict fail, ex: {0}".format(str(ex)), file=sys.stderr)
        pass
    return data


def dict2json(data, encoding='utf8'):
    json_str = ''
    try:
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False).encode(encoding=encoding)
    except Exception as ex:
        # works in python3
        # print("dict2json fail, ex: {0}".format(str(ex)), file=sys.stderr)
        pass

    return json_str


def get_project_root():
    """Returns project root folder."""
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SRC_DIR)
    return ROOT_DIR


# pySpark


def save_rdd_hdfs(rdd, path):
    path_tmp = path.strip('\\/') + ".tmp"
    cmd = 'hadoop fs -rm -r ' + path_tmp
    os.system(cmd)
    rdd.saveAsTextFile(path_tmp)
    cmd = 'hadoop fs -rm -r %s; hadoop fs -mv %s %s' % (path, path_tmp, path)
    os.system(cmd)
    # NOTICE("save rdd to %s success"%path)


def pro_init():
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # print("ROOT_DIR:" + str(ROOT_DIR))

    source_dirs = ['conf', 'src', 'common']
    for i in source_dirs:
        py_dir = str(ROOT_DIR) + '/' + i
        if os.path.exists(py_dir):
            sys.path.append(py_dir)
    # print("sys path: " + str(sys.path))


def get_current_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_argument_parser():
    """
    flags, unused_flags = parser.parse_known_args()
    flags.log_dir
    :return:
    """
    current_path = get_current_path()

    print("os.path.realpath: {}".format(os.path.realpath(sys.argv[0])))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    return parser


pro_init()


