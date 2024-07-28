import * as tf from '@tensorflow/tfjs-node'

import { generateLinearData } from '../src/js/patterns.js'

const model = tf.sequential()

// Add layers
model.add(tf.layers.dense({ units: 4, inputShape: [2], activation: 'relu' }))
model.add(tf.layers.dense({ units: 3, activation: 'relu' }))
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }))

// Compile the model
model.compile({
  optimizer: tf.train.adam(0.01),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
})

// Generate data
const classifier = (x, y) => (x > y ? 1 : 0)
const [xs, ys] = generateLinearData(1000, classifier)
const [xst, yst] = generateLinearData(10, classifier)
const xsT = tf.tensor2d(xst)

const xsTensor = tf.tensor2d(xs)
const ysTensor = tf.tensor2d(ys.map((v) => [v]))

// Train the model
await model.fit(xsTensor, ysTensor, {
  epochs: 10,
  batchSize: 5,
  verbose: 1,
  shuffle: true,
})

const predictions = model.predict(xsT)
const predictedLabels = predictions
  .arraySync()
  .map((pred) => (pred[0] > 0.5 ? 1 : 0))

let correct = 0
predictedLabels.forEach((pred, index) => {
  if (pred === yst[index]) {
    correct += 1
  }
})

const accuracy = correct / yst.length
console.info(
  `Accuracy: ${new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
  }).format(accuracy)}`,
)
