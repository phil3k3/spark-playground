package at.datasciencelabs.analysis.finance;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class ClassifyExpenses {

    public static void main(String[] args) {

        SparkSession sparkSession = SparkSession.builder().master("local").appName("expenses-classify").getOrCreate();
        sparkSession.conf().set("spark.driver.memory", "4g");
        sparkSession.conf().set("spark.executor.memory", "4g");

        Dataset<Row> dataSet = sparkSession.read().option("header", "true").option("delimiter", ";").csv("data2_annotated.csv");
        dataSet.printSchema();

        for (Row row : dataSet.takeAsList(5)) {
            System.out.println(row.getString(0));
        }

        // then we need a tokenizer

        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
        Dataset<Row> words = tokenizer.transform(dataSet);

        // now we need a hashing tf

        HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100000);
        Dataset<Row> features = hashingTF.transform(words);

        IDF idfModel = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel1 = idfModel.fit(features);

        Dataset<Row> features2 = idfModel1.transform(features);

        features2.printSchema();
        JavaRDD<LabeledPoint> labeledPointJavaRDD = features2.javaRDD().map((Function<Row, LabeledPoint>) v1 -> {
            SparseVector featureVector = v1.getAs(10);
            org.apache.spark.mllib.linalg.SparseVector newSparseVector = new org.apache.spark.mllib.linalg.SparseVector(
                    featureVector.size(),
                    featureVector.indices(),
                    featureVector.values());
            return new LabeledPoint((double)v1.getString(0).hashCode(), newSparseVector);
        });

        JavaRDD<LabeledPoint>[] result = labeledPointJavaRDD.randomSplit(new double[]{0.6,0.4});
        final NaiveBayesModel model = NaiveBayes.train(result[0].rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel = result[1].mapToPair((PairFunction<LabeledPoint, Double, Double>) labeledPoint -> new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label()));

        double accuracy = ((double)(predictionAndLabel.filter((Function<Tuple2<Double, Double>, Boolean>) v1 -> v1._1().equals(v1._2())).count())) / (double)result[1].count();

        System.out.println(accuracy);
    }
}
