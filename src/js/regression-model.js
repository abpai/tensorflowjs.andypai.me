import * as tf from '@tensorflow/tfjs'
import range from './range.js'

const randomNumber = (min, max) => Math.random() * (max - min) + min

export const generateTrainingData = (
  xRange,
  fn = (x) => x,
  { n = 300 } = {},
) => {
  const xs = []
  const ys = []
  const { min, max } = xRange

  range(0, n).forEach(() => {
    const x = randomNumber(min, max)
    xs.push([x])
    ys.push(fn(x))
  })

  return [xs, ys]
}

export const buildModel = () => {
  // Create a Sequential model
  const model = tf.sequential()

  // Add layers
  model.add(tf.layers.dense({ units: 10, inputShape: [1], activation: 'tanh' }))
  model.add(tf.layers.dense({ units: 10, activation: 'tanh' }))
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }))

  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.05),
    loss: 'meanSquaredError',
  })

  return model
}
