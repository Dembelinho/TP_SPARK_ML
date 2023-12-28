package org.example;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KmeansAPP {
    public static void main(String[] args) {
        SparkSession sparkSession=SparkSession.builder().appName("TP ML SPARK").master("local[*]").getOrCreate();
        Dataset<Row> dataFrame = sparkSession.read()
                .option("inferSchema", true)
                .option("header", true).csv("Mall_Customers.csv");

        dataFrame.printSchema();
        dataFrame.show(5);

        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"})
                .setOutputCol("Features");
        Dataset<Row> Assembled_dataset = assembler.transform(dataFrame);
        MinMaxScaler scaler = new MinMaxScaler().setInputCol("Features").setOutputCol("NormalizedFeatures");
        Dataset<Row> normalizedDataset = scaler.fit(Assembled_dataset).transform(Assembled_dataset);

        // Définition du modèle K-means, entraînement du modèle et prédiction
        KMeans kMeans = new KMeans().setK(3).setFeaturesCol("NormalizedFeatures").setPredictionCol("Cluster");
        KMeansModel model = kMeans.fit(normalizedDataset);
        Dataset<Row> predictions = model.transform(normalizedDataset);
        predictions.show(200);
    }
}
