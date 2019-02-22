import sys
import pyspark

# reload(sys)
# sys.setdefaultencoding('utf-8')

class SparkContext(pyspark.SparkContext):

    def __init__(self, master = None, appName = None, sparkHome = None, pyFiles = None, environment = None, batchSize = 0, serializer = pyspark.PickleSerializer(), conf = None, gateway = None, jsc = None): pyspark.SparkContext.__init__(
                self,
                master = master,
                appName = appName,
                sparkHome = sparkHome,
                pyFiles = pyFiles,
                environment = environment,
                batchSize = batchSize,
                serializer = serializer,
                conf = conf,
                gateway = gateway,
                jsc = jsc)

    def textFiles(self, dirs):
        hadoopConf = {"mapreduce.input.fileinputformat.inputdir" : ",".join(dirs), "mapreduce.input.fileinputformat.input.dir.recursive" : "true"}
        pair = self.hadoopRDD(
                inputFormatClass = "org.apache.hadoop.mapred.TextInputFormat",
                keyClass = "org.apache.hadoop.io.LongWritable",
                valueClass = "org.apache.hadoop.io.Text",
                conf = hadoopConf
                )
        text = pair.map(lambda pair : pair[1])
        return text