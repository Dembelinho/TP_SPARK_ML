package org.example;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
public class LinearRegressionAPP {
    public static void main(String[] args) {
        SparkSession sparkSession=SparkSession.builder().appName("TP ML SPARK").master("local[*]").getOrCreate();
        Dataset<Row> dataFrame = sparkSession.read()
                .option("inferSchema", true)
                .option("header", true).csv("Advertising.csv");
        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{"TV", "Radio", "Newspaper"})
                .setOutputCol("Features");

        Dataset<Row> dataset = assembler.transform(dataFrame);
        Dataset<Row>[] split = dataset.randomSplit(new double[]{0.8, 0.2}, 1234);
        Dataset<Row> train = split[0];
        Dataset<Row> test = split[1];

        LinearRegression linearRegression= new LinearRegression().setLabelCol("Sales").setFeaturesCol("Features");
        LinearRegressionModel regressionModel = linearRegression.fit(train);
        Dataset<Row> pred = regressionModel.transform(test);

        pred.show();
        System.out.println("Intercept= "+regressionModel.intercept());
        System.out.println("Coeff= "+regressionModel.coefficients());
    }
}