package at.datasciencelabs.analysis.finance;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import static org.apache.spark.sql.functions.*;
import scala.Tuple2;

import java.text.NumberFormat;
import java.util.Locale;

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

        sparkSession.udf().register("fromGermanDoubleString", new UDF1<String, Double>() {
            @Override
            public Double call(String s) throws Exception {
                NumberFormat numberFormat = NumberFormat.getNumberInstance(Locale.GERMAN);
                return numberFormat.parse(s).doubleValue();
            }
        }, DataTypes.DoubleType);

        sparkSession.udf().register("vectorFromColumn", new UDF1<Double, Vector>() {
            @Override
            public Vector call(Double aDouble) throws Exception {
                return Vectors.dense(aDouble);
            }
        }, new VectorUDT());

        Dataset<Row> features2WithDataTypeChanged = features2.withColumn("amountDouble", callUDF("fromGermanDoubleString", features2.col("amount")));
        features2WithDataTypeChanged.printSchema();

        for(Row row : features2WithDataTypeChanged.takeAsList(5)) {
            Double doubleValue = row.getAs("amountDouble");
            System.out.println(doubleValue);
        }
        Dataset<Row> dataSetOnlyExpenses = features2WithDataTypeChanged.filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row value) throws Exception {
                Double amount = value.getAs("amountDouble");
                return amount < 0;
            }
        });
        Dataset<Row> dataSetOnlyExpenses2 = dataSetOnlyExpenses.withColumn("vectorizedAmount", callUDF("vectorFromColumn", dataSetOnlyExpenses.col("amountDouble")));

        MinMaxScaler scaler = new MinMaxScaler().setInputCol("vectorizedAmount").setOutputCol("amountNormalized").setMin(0.0).setMax(1.0);

        Dataset<Row> transform = scaler.fit(dataSetOnlyExpenses2).transform(dataSetOnlyExpenses2);
        transform.show();

        JavaRDD<LabeledPoint> labeledPointJavaRDDWithExpenses = transform.javaRDD().map((Function<Row, LabeledPoint>) v1 -> {

            Vector amount = v1.getAs("amountNormalized");

            SparseVector featureVector = v1.getAs(10);
            int[] indices = featureVector.indices();
            int[] newIndices = new int[indices.length+1];
            System.arraycopy(indices, 0, newIndices, 0, indices.length);
            newIndices[newIndices.length-1] = featureVector.size();

            double[] values = featureVector.values();
            double[] newValues = new double[values.length+1];
            System.arraycopy(values, 0, newValues, 0, values.length);
            newValues[newValues.length-1] = amount.toArray()[0];

            org.apache.spark.mllib.linalg.SparseVector newSparseVector = new org.apache.spark.mllib.linalg.SparseVector(
                    featureVector.size(),
                    featureVector.indices(),
                    featureVector.values());

            return new LabeledPoint((double)v1.getString(0).hashCode(), newSparseVector);
        });


        JavaRDD<LabeledPoint>[] result = labeledPointJavaRDDWithExpenses.randomSplit(new double[]{0.6,0.4}, 11L);
        final NaiveBayesModel model = NaiveBayes.train(result[0].rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel = result[1].mapToPair((PairFunction<LabeledPoint, Double, Double>) labeledPoint -> new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label()));

        double accuracy = ((double)(predictionAndLabel.filter((Function<Tuple2<Double, Double>, Boolean>) v1 -> v1._1().equals(v1._2())).count())) / (double)result[1].count();

        System.out.println(accuracy);
    }
}
