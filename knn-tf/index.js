require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
  return (
    features
      .sub(mean)                 // Scaling (standartization).
      .div(variance.pow(0.5))    //
      .sub(scaledPrediction)     // Calculating range between prediction point and all other points in test set, using some perverted Pythagorean theorem application.
      .pow(2)                    //
      .sum(1)                    //
      .pow(0.5)                  //
      .expandDims(1)             //
      .concat(labels, 1)         //
      .unstack()
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living", "condition"],
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i)=> {

  const predictedResult = knn(features, labels, tf.tensor(testPoint), 8);
  const err = (testLabels[i][0] - predictedResult) / testLabels[i][0];

  console.log('Error : ', err * 100, '%.');
})
