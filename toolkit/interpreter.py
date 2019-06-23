# -*- coding: utf-8 -*-
""" 离线拉链 迁移盘古
    低耦合、高复用 之典范
    author Ethan Jo
    tuixing.zx@alibab-inc.com
    墙裂推荐使用python3
"""
import os
import sys
import json
import types
import base64


def formate2print(db_key, value_list):
    """
    eg: index_name=kExp_9999`index_data=111111,7.0#2222222,2.3#333333,1.3
    :param db_key: which can be index for DB
    :param value_list: list of (item, score) tuple
    :return:
    """
    tmp = "index_name={}`index_data={}"
    kv_str_list = ["{},{}".format(i[0], i[1]) for i in value_list]
    index_data = '#'.join(kv_str_list)

    print(tmp.format(db_key, index_data))


class BaseTranslator(object):
    """Abstract SingleKvChain

    # Properties
    # Methods
    # Class Methods
    # Internal methods:

    """

    def __init__(self, **kwargs):

        self.prefix = []
        self.language = []
        self.private_key_b64 = False

        # These lists will be filled via successive calls
        self._private_key = ""
        self._item_list = []

        # These properties should be set by the user via keyword arguments.
        # note that 'prefix', 'language' and 'private_key_b64'
        # allowed_kwargs = {
        #                   'prefix',
        #                   'language',
        #                   'private_key_b64',
        #                   }
        # for kwarg in kwargs:
        #     if kwarg not in allowed_kwargs:
        #         raise TypeError('Keyword argument not understood:', kwarg)

        if 'prefix' in kwargs:
            if isinstance(kwargs['prefix'], str):
                self.prefix.append(kwargs['prefix'])
            elif isinstance(kwargs['prefix'], (tuple, list)):
                self.prefix = list(kwargs['prefix'])
        if 'language' in kwargs:
            if isinstance(kwargs['language'], str):
                self.language.append(kwargs['language'])
            elif isinstance(kwargs['language'], (tuple, list)):
                self.language = list(kwargs['language'])

        if 'private_key_b64' in kwargs and isinstance(kwargs['private_key_b64'], bool):
            # self.private_key = base64.b64encode(str(std_obj[0]).encode()).decode('utf-8')
            self.private_key_b64 = kwargs['private_key_b64']

    def call(self, input_line):
        """
        This should return single tuple
        The tuple should be (private_key, item_sore tuple list)
        :param input_line:
        :return:
        """
        o = eval(input_line)
        # return o[0], o[1]
        return o[0], o[1]

    def yield_dbkey_items(self):
        if self.prefix is None:
            self.prefix = []
        if self.language is None:
            self.language = []

        if isinstance(self.prefix, str):
            tmp = self.prefix
            self.prefix = []
            self.prefix.append(tmp)
        if isinstance(self.language, str):
            tmp = self.language
            self.language = []
            self.language.append(tmp)

        if len(self.prefix) == 0 and len(self.language) == 0:
            # print(self.prefix, self.language)
            key = self._private_key
            yield key, self._item_list

        elif len(self.prefix) > 0 and len(self.language) == 0:
            for i in self.prefix:
                key = BaseTranslator.key_maker(prefix=i, private_key=self._private_key)
                if key is None:
                    yield
                yield key, self._item_list

        elif len(self.prefix) > 0 and len(self.language) > 0:
            for i in self.prefix:
                for j in self.language:
                    key = BaseTranslator.key_maker(prefix=i, language=j, private_key=self._private_key)
                    yield key, self._item_list

    def __call__(self, input_line):
        if not isinstance(input_line, (str, tuple, dict)):
            return
        call_result = self.call(input_line)
        if call_result is None:
            return

        self._private_key, self._item_list = call_result

        if self.private_key_b64:
            self._private_key = base64.b64encode(str(self._private_key).encode()).decode('utf-8')

        generator = self.yield_dbkey_items()

        if isinstance(generator, tuple) and len(generator) == 2:
            formate2print(generator[0], generator[1])
        elif isinstance(generator, types.GeneratorType):
            for i in generator:
                formate2print(i[0], i[1])

    @classmethod
    def key_maker(cls, prefix, language=None, private_key=None, private_key_b64=False):
        if private_key is None:
            return
        # if private_key_b64:
        #     private_key = base64.b64encode(str(private_key).encode()).decode('utf-8')
        if prefix is not None and private_key is not None and language is not None:
            return "{}_{}_{}".format(prefix, language, private_key)
        elif prefix is not None and private_key is not None and language is None:
            return "{}_{}".format(prefix, private_key)


if __name__ == '__main__':
    print("test")
    # translator = BaseTranslator(prefix=["SIG", "HAHA"], language=["indonesian", "indi"], private_key_b64=False)
    translator = BaseTranslator()

    for line in open('kRealtimeContext.input'):
    # for line in open('kSig.input'):
        # translator = BaseTranslator(input_data=line, callback=kSig_to_std_obj, private_key_b64=True)
        # print(line)
        translator(input_line=line)
