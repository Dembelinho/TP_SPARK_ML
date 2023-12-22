package org.example;

import org.apache.spark.sql.SparkSession;

public class KmeansAPP {
    public static void main(String[] args) {
        SparkSession sparkSession=SparkSession.builder().appName("TP ML SPARK").master("local[*]").getOrCreate();
    }
}
