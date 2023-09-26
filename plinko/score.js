"use strict";
const outputs = [];
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
    outputs.push([dropPosition, bounciness, size, bucketLabel]);
}
/** Absolute distance from the prediction point to the point where the ball was actually dropped */
function distance(pointA, pointB) {
    return (pointA.reduce((acc, point, index) => {
        acc = acc + (point - pointB[index]) ** 2;
        return acc;
    }, 0) ** 0.5);
}
/** Splitting all the results to the test set and training set */
function splitDataset(data, testCount) {
    const shuffled = window._.shuffle(data);
    const testSet = shuffled.slice(0, testCount);
    const trainingSet = shuffled.slice(testCount);
    return [testSet, trainingSet];
}
/** Prediction algorythm, k - nearest neighbour */
function knn(data, point, k) {
    const res = Object.entries({
        ...[...data]
            .map((row) => {
            const a = [distance(row.slice(0, -1), point), row.at(-1)];
            return a;
        })
            .sort((a, b) => a[0] - b[0])
            .slice(0, k)
            .reduce((acc, el) => {
            const fieldName = el[1];
            fieldName in acc ? (acc[fieldName] += 1) : (acc[fieldName] = 1);
            return acc;
        }, {}),
    }).sort((a, b) => b[1] - a[1]);
    return res[0][0];
}
function runAnalysis() {
    const testSetSize = 100;
    const k = 10;
    for (let feature = 0; feature < 3; feature += 1) {
        const data = outputs.map(row => [row[feature], row.at(-1)]);
        const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
        const accuracy = (testSet.filter((testPoint) => {
            const bucket = knn(trainingSet, testPoint.slice(0, -1), k);
            return Number(bucket) === testPoint.at(-1);
        }).length /
            testSetSize) *
            100;
        console.log("Accuracy is " + Math.ceil(accuracy) + " %, feature is " + feature);
    }
}
function minMax(data, featureCount) {
    const clonedData = [...data].map((row) => [...row]);
    for (let i = 0; i < featureCount; i += 1) {
        const column = clonedData.map((row) => row[i]);
        const sortedColumn = [...column].sort((a, b) => a - b);
        const min = sortedColumn[0];
        const max = sortedColumn.at(-1);
        for (let j = 0; j < clonedData.length; j += 1) {
            clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
        }
    }
    return clonedData;
}
