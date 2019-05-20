import urllib.parse
import base64
import argparse


def urlencode(str):
    return urllib.parse.quote(str)


def urldecode(str):
    return urllib.parse.unquote(str)


def base64encode(str):
    return base64.b64encode(str)


def base64decode(str):
    return base64.b64decode(str)


def convert_encoded_filter_reason(str):
    filter_reason_list = list()
    for i in urldecode(str).split("|"):
        filter_reason_list.append(base64decode(i).decode("utf-8"))
    return filter_reason_list


def test():
    str = '{"name": "Kinkin"}'
    encoded = urlencode(str)
    print(encoded)  # '%7B%22name%22%3A%20%22Kinkin%22%7D'
    decoded = urldecode(encoded)
    print(decoded)  # '{"name": "Kinkin"}'


if __name__ == '__main__':
    #
    a = "MDAw%7CMDAx%7CMDAy%7CMDAz%7CMDA0%7CMDA1%7CMDA2%7CMDA3%7CMDA4%7CMDEw%7CMDEx%7CMDEy%7CMDEz%7CMDE0%7CMDE1%7CMDE2%7CMDE3%7CMDE4%7CMDIw%7CMDIx%7CMDIz%7CMDI1%7CMDI2%7CMDI5"

    #
    final_category_id_intersect = "MDA4%7CMDMy%7CMDA0MDA2%7CMDE3%7CMDI1MDAy%7CMDA0MDA1%7CMDE0%7CMDI1MDAz%7CMDE2MDAx%7CMDE1%7CMDI5%7CMDE4%7CMDI2%7CMDA1%7CMDE0MDA1%7CMDE2MDAy%7CMDI1MDAx%7CMDI4%7CMDA1MDEx%7CMDA0%7CMDA4MDAx%7CMDA3%7CMDA2MDAx%7CMDA2%7CMDIz%7CMDAyMDAx%7CMDIy%7CMDAx%7CMDExMDAx%7CMDIx%7CMDAxMDAw%7CMDA0MDAz%7CMDEy%7CMDIw%7CMDAz%7CMDEz%7CMDI1MDA2%7CMDAy%7CMDI1MDA4%7CMDA0MDAx%7CMDEw%7CMDI1MDA3%7CMDMw%7CMDEx%7CMDI1MDA0%7CMDA5%7CMDE0MDAx%7CMDEyMDA3%7CMDA0MDA3%7CMDE2%7CMDI1MDA1"

    print(convert_encoded_filter_reason(final_category_id_intersect))
