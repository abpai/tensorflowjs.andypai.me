import { evaluate } from 'mathjs'
import * as tf from '@tensorflow/tfjs-node'

// Initial learning rate
const initialLearningRate = 0.001

// Create a Sequential model
const model = tf.sequential()

// Add layers
model.add(tf.layers.dense({ units: 4, inputShape: [3], activation: 'relu' }))
model.add(tf.layers.dense({ units: 3, activation: 'relu' }))
model.add(tf.layers.dense({ units: 1, activation: 'linear' }))

// Compile the model
model.compile({
  optimizer: tf.train.adam(0.01),
  loss: 'meanSquaredError',
})

// Create an SGD optimizer
const optimizer = tf.train.sgd(initialLearningRate)

// Compile the model
model.compile({
  optimizer,
  loss: 'meanSquaredError',
})

const expression = '2x + 3y - z'
const fn = (x, y, z) => evaluate(expression, { x, y, z })

const dataset = Array.from({ length: 1000 }, () => {
  const inputs = [Math.random() * 10, Math.random() * 10, Math.random() * 10]
  return {
    inputs,
    output: fn(...inputs),
  }
})

const xs = tf.tensor2d(dataset.map(({ inputs }) => inputs))
const ys = tf.tensor2d(dataset.map(({ output }) => [output]))

await model.fit(xs, ys, {
  epochs: 50,
  verbose: 1,
})

// Train the model and2 then make a prediction
const input = [2.0, 3.0, -1.0]
const prediction = model.predict(tf.tensor2d([input]))
console.info(`
  Prediction: ${prediction.dataSync()}
  Target: ${fn(...input)}
`)
