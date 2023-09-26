import tf from "@tensorflow/tfjs";
import _ from "lodash";

class LinearRegression {
  constructor(features, labels, options) {
    this.labels = tf.tensor(labels);
    this.features = this.#processFeatures(features);
    this.options = { learningRate: 0.1, iterations: 1000, ...options };
    this.weights = tf.zeros([2, 1]);
  }

  train() {
    for (let i = 0; i < this.options.iterations; i += 1) {
      this.gradientDescent();
    }
  }

  test(testFeatures, testLabels) {
    testLabels = tf.tensor(testLabels);
    testFeatures = this.#processFeatures(testFeatures);
    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels.sub(predictions).pow(2).sum().get();
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    return 1 - res / tot;
  }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    const { mean, variance } = tf.moments(this.features, 0);
    console.log(mean.print(), variance.print());
  }

  #processFeatures(features){
    const featuresTensor = tf.tensor(features);
    return tf.ones([featuresTensor.shape[0], 1]).concat(featuresTensor, 1);

  }

  // gradientDescent() {
  //   const currentGuessForMPG = this.features.map(
  //     (row) => this.m * row[0] + this.b
  //   );

  //   const mSlope =
  //     (_.sum(
  //       currentGuessForMPG.map(
  //         (guess, i) => this.features[i][0] * -1 * (this.labels[i][0] - guess)
  //       )
  //     ) *
  //       2) /
  //     this.features.length;

  //   const bSlope =
  //     (_.sum(currentGuessForMPG.map((guess, i) => guess - this.labels[i][0])) *
  //       2) /
  //     this.features.length;

  //   this.m = this.m - mSlope * this.options.learningRate;
  //   this.b = this.b - bSlope * this.options.learningRate;
  // }
}

export default LinearRegression;
