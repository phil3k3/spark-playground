package at.datasciencelabs.analysis.finance;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class Analyzer {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("analysis").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> csvFile = sc.textFile("data.csv");

        expensesByMonth(csvFile);
        expensesByCategory(csvFile);
    }


    private static void expensesByCategory(JavaRDD<String> csvFile) {
        JavaPairRDD<String, Integer> transactions = csvFile.mapToPair((PairFunction<String, String, Integer>) s -> {
            String[] splitLine = s.split(";");
            String category = splitLine[1].split(" ")[0];
            int expenseCents = Integer.parseInt(splitLine[3].replace(",",""));
            return new Tuple2<>(category, expenseCents);
        });

        JavaPairRDD<String, Integer> expensesPerCategory = transactions.reduceByKey((a, b) -> a+b).sortByKey();


        List<Tuple2<Integer, String>> expenses = expensesPerCategory.mapToPair((PairFunction<Tuple2<String, Integer>, Integer, String>) Tuple2::swap
        ).sortByKey().collect();

        for(Tuple2<Integer, String> expenseEntry : expenses) {
            System.out.println(expenseEntry._2() + " " + ((float)expenseEntry._1())/100);
        }
    }

    private static void expensesByMonth(JavaRDD<String> csvFile) {
        JavaPairRDD<LocalDate, Integer> transactions = csvFile.mapToPair((PairFunction<String, LocalDate, Integer>) s -> {
            String[] splitLine = s.split(";");
            LocalDate date = LocalDate.parse(splitLine[0], DateTimeFormatter.ofPattern("dd.MM.yyyy"));
            LocalDate beginningOfMonth = date.withDayOfMonth(1);
            int expenseCents = Integer.parseInt(splitLine[3].replace(",",""));
            return new Tuple2<>(beginningOfMonth, expenseCents);
        });

        JavaPairRDD<LocalDate, Integer> expensesPerMonth = transactions.reduceByKey((a, b) -> a+b).sortByKey();

        List<Tuple2<LocalDate, Integer>> expenses = expensesPerMonth.collect();

        for(Tuple2<LocalDate, Integer> expenseEntry : expenses) {
            System.out.println(expenseEntry._1() + " " + ((float)expenseEntry._2())/100);
        }
    }
}
