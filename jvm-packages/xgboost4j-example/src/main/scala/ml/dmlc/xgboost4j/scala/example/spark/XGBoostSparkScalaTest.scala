package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.Booster
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by vsanjekar on 8/17/17.
  */
object XGBoostSparkScalaTest {

  def main(args: Array[String]) {
    /*
    println(args.mkString(" "))
    if (args.length != 5) {
      println(
        "usage: program num_of_rounds num_workers training_path test_path model_path")
      sys.exit(1)
    }
    val inputTrainPath = args(0)
    val inputTestPath = args(1)
    val outputPath = args(2)
    val numRound = args(3).toInt
    val numWorkers = args(4).toInt
    */

    val inputTrainPath = "xgboost4j-example/data/liver-disorders"
    val inputTestPath = "xgboost4j-example/data/liver-disorders.t"
    val outputPath = "xgboost4j-example/data/output"
    val numRound = 10
    val numWorkers = 10

    // spark initialization
    val sparkConf = new SparkConf()
      .setAppName("Test XGBoost")
      .setMaster("local[1]")
    val sc = new SparkContext(sparkConf)

    // data
    val trainRDD = MLUtils.loadLibSVMFile(sc, inputTrainPath)
    // val testRDD = MLUtils.loadLibSVMFile(sc, inputTestPath)

    trainRDD.collect().foreach(println)

    // parameters
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap

    // train
    val model = XGBoost.train(trainRDD, paramMap, numRound, numWorkers, useExternalMemory = true)
    model.saveModelAsHadoopFile(outputPath)(sc)
  }
}
