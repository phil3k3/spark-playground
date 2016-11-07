import com.google.common.base.Joiner;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Created by phil3k on 01.11.16.
 */
public class HashingTFTest {

    @Test
    public void testHashingTF(){

        List<Row> data = Arrays.asList(
                RowFactory.create("food", "MERKUR Wien abc"),
                RowFactory.create("car", "OMV Tankstelle"),
                RowFactory.create("food", "MERKUR Markt"),
                RowFactory.create("food", "Vapiano"),
                RowFactory.create("car", "OMV Vapiano Merkur Merkur")
        );

        int numFeatures = 100;

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.StringType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
        });
        SparkSession sparkSession = SparkSession.builder().master("local").appName("analysis").getOrCreate();
        Dataset<Row> sentenceData = sparkSession.createDataFrame(data, schema);
        execute(sentenceData, numFeatures);
    }

    @Test
    public void testWithActualData() {
        SparkSession sparkSession = SparkSession.builder().master("local").appName("analysis").getOrCreate();
        Dataset<Row> dataSet = sparkSession.read().option("header", "true").option("delimiter", ";").csv("simple.csv");
        dataSet.printSchema();
        execute(dataSet, 20000);
    }


    private void execute(Dataset<Row> sentenceData, int numFeatures) {

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        for (Row r : wordsData.select("sentence", "words").takeAsList(5)) {
            String sentence = r.getAs(0);
            List<String> words = r.getList(1);
            System.out.println(sentence);
            System.out.println(Joiner.on(",").join(words));
        }


        HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures);
        Dataset<Row> features = hashingTF.transform(wordsData);

        for (Row r : features.select("words", "rawFeatures").takeAsList(5)) {
            List<String> words = r.getList(0);
            SparseVector featureVector = r.getAs(1);
            System.out.println(featureVector);
            System.out.println(Joiner.on(",").join(words));
        }

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(features);
        Dataset<Row> finalFeatures = idfModel.transform(features);
        for (Row r : finalFeatures.select("features", "label").takeAsList(5)) {
            SparseVector featureVector = r.getAs(0);
            String label = r.getString(1);
            System.out.println(featureVector);
            System.out.println(label);
        }
    }


}
