import "@tensorflow/tfjs-node";
import tf from "@tensorflow/tfjs";
import LinearRegression from "./linear-regression.mjs";
import loadCSV from "./load-csv.js";

const { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);
console.log(r2);
