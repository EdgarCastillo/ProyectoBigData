import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline

val df = spark.read.option("inferSchema","true").csv("IRIS.csv").toDF("SepalLength", "SepalWidth", "PetalLength", "PetalWidth","class")
val newcol = when($"class".contains("Iris-setosa"), 1.0).otherwise(when($"class".contains("Iris-virginica"), 3.0).otherwise(2.0))
val newdf = df.withColumn("etiqueta", newcol)

val assembler = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth","etiqueta")).setOutputCol("features")
val features = assembler.transform(newdf)


val bkm = new BisectingKMeans().setK(3).setSeed(1)
val model = bkm.fit(features)

val WSSE = model.computeCost(features)
println(s"Within set sum of Squared Errors = $WSSE")

// Shows the result.
println("Cluster Centers: ")
val centers = model.clusterCenters
centers.foreach(println)
