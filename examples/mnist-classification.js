import * as tf from '@tensorflow/tfjs-node'
import datasets from './datasets/index.js'

const dataset = datasets.read('mnist', 'train')
const [xs, ys] = dataset.reduce(
  (prev, data) => {
    const { image, label } = data
    prev[0].push(image)
    prev[1].push(label)
    return prev
  },
  [[], []],
)
const uniqueLabels = 10

// Specify the architecture
const buildModel = () => {
  const model = tf.sequential()
  model.add(
    tf.layers.dense({ inputShape: [14 * 14], units: 20, activation: 'relu' }),
  )
  model.add(tf.layers.dense({ units: 20, activation: 'relu' }))
  model.add(tf.layers.dense({ units: uniqueLabels, activation: 'softmax' }))

  return model
}

const model = buildModel()

model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
})

const oneHotEncode = (label, numClasses) => {
  const encoding = Array(numClasses).fill(0)
  encoding[label] = 1
  return encoding
}

const xsTensor = tf.tensor2d(xs)
const ysTensor = tf.tensor2d(ys.map((y) => oneHotEncode(y, uniqueLabels)))
model
  .fit(xsTensor, ysTensor, {
    epochs: 100,
    batchSize: 25,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.info(
          `Epoch: ${epoch + 1}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`,
        )
      },
    },
  })
  .then(() => {
    const prediction = model.predict(tf.tensor2d([xs[0]]))
    console.info(`Prediction: ${prediction.argMax(-1).dataSync()[0]}`) // Output: 1
    console.info(`Actual: ${ys[0]}`)
  })
