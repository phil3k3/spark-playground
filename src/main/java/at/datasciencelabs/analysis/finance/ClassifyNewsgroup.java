package at.datasciencelabs.analysis.finance;

import com.google.common.base.Joiner;
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

import java.util.List;

/**
 * Created by phil3k on 07.11.16.
 */
public class ClassifyNewsgroup {

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().master("local").appName("analysis").getOrCreate();

        JavaRDD<Tuple2<String, String>> content = sparkSession.sparkContext().wholeTextFiles("20_newsgroup/*", 1).toJavaRDD();
        JavaRDD<FileBean> files = content.map((Function<Tuple2<String, String>, FileBean>) v1 -> new FileBean(v1._1, v1._2));

        Dataset<Row> sentences = sparkSession.createDataFrame(files, FileBean.class);
        sentences.printSchema();
        System.out.println(sentences.count());

        Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentences);

        for (Row r : wordsData.select("words").takeAsList(5)) {
            List<String> words = r.getList(0);
            System.out.println(Joiner.on(",").join(words));
        }

        // num features must be the distinct amount of words in the corpus?
        HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000000);
        Dataset<Row> features = hashingTF.transform(wordsData);

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(features);

        Dataset<Row> idfFeatures = idfModel.transform(features);

        idfFeatures.printSchema();


        for (Row row : idfFeatures.takeAsList(5)) {
            System.out.println(row.getString(0));
        }

        JavaRDD<LabeledPoint> labeledPointJavaRDD = idfFeatures.javaRDD().map((Function<Row, LabeledPoint>) s -> {
            double label = new NewsGroupLabelConversion().convert(s.getString(0));
            SparseVector featureVector = s.getAs(4);
            org.apache.spark.mllib.linalg.SparseVector newSparseVector = new org.apache.spark.mllib.linalg.SparseVector(
                    featureVector.size(),
                    featureVector.indices(),
                    featureVector.values());
            return new LabeledPoint(label, newSparseVector);
        });

        JavaRDD<LabeledPoint>[] result = labeledPointJavaRDD.randomSplit(new double[]{0.6, 0.4});
        final NaiveBayesModel model = NaiveBayes.train(result[0].rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel = result[1].mapToPair((PairFunction<LabeledPoint, Double, Double>) labeledPoint -> new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label()));

        double accuracy = ((double)(predictionAndLabel.filter((Function<Tuple2<Double, Double>, Boolean>) v1 -> v1._1().equals(v1._2())).count())) / (double)result[1].count();

        System.out.println(accuracy);
    }
}
