# encoding = utf-8
import pyspark
import numpy as np
import pandas as pd
import extend_pyspark
import argparse
import conf


def GetSparkContext():
    conf = pyspark.SparkConf()
    conf.setAppName("analysis_nochain_seed")
    conf.set("spark.executor.memory", "1G")
    conf.set("spark.kryoserializer.buffer.max","512M")
    sc = extend_pyspark.SparkContext(conf=conf)
    return sc


def analysis(sc, nochain_seed_path="/home/datamining/tuixing.zx/seed_increase_evaluation"):
    # read nochain seed
    nochain_seed = sc.textFile(nochain_seed_path)

    # read news_reco
    item_infos = sc.textFile(conf.HDFS_BASE_ITEM_INFOS, use_unicode=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="analysis for nochain seed")
    parser.add_argument("-r", "--report_path", type=str, default="./report/report_tmp.txt",
                        help="Path for report")
    parser.add_argument("-s", "--nochain_seed", type=str, default="./data/nochain_seed.txt")
    sc = GetSparkContext()

    analysis(sc)

    sc.stop(sc)