import { join } from 'path'
import * as tf from '@tensorflow/tfjs-node'

import datasets from './datasets/index.js'

const dataset = datasets.read('mnist', 'train')

const LATENT_SIZE = 49
const __dirname = import.meta.dirname

// Define the model architecture
const buildModel = () => {
  const encoder = tf.sequential()
  encoder.add(tf.layers.dense({ units: 196, inputShape: [196] }))
  encoder.add(tf.layers.leakyReLU())
  encoder.add(tf.layers.dropout({ rate: 0.5 }))

  encoder.add(tf.layers.dense({ units: LATENT_SIZE * 2 }))
  encoder.add(tf.layers.leakyReLU())
  encoder.add(tf.layers.dropout({ rate: 0.5 }))

  encoder.add(tf.layers.dense({ units: LATENT_SIZE }))
  encoder.add(tf.layers.leakyReLU())

  const decoder = tf.sequential()
  decoder.add(
    tf.layers.dense({ units: LATENT_SIZE, inputShape: [LATENT_SIZE] }),
  )
  decoder.add(tf.layers.leakyReLU())
  decoder.add(tf.layers.dropout({ rate: 0.5 }))

  decoder.add(tf.layers.dense({ units: LATENT_SIZE * 2 }))
  decoder.add(tf.layers.leakyReLU())
  decoder.add(tf.layers.dropout({ rate: 0.5 }))

  decoder.add(tf.layers.dense({ units: 196, activation: 'sigmoid' }))

  const input = tf.input({ shape: [196] })
  const encoded = encoder.apply(input)
  const decoded = decoder.apply(encoded)
  const autoencoder = tf.model({ inputs: input, outputs: decoded })

  return {
    autoencoder,
    encoder,
    decoder,
  }
}

const fitAndSave = async () => {
  const { autoencoder, encoder, decoder } = buildModel()

  autoencoder.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
  })

  const xs = dataset.map(({ image }) => image)
  console.info(`Training on ${xs.length} images`)

  const xsTensor = tf.tensor2d(xs)
  autoencoder
    .fit(xsTensor, xsTensor, {
      epochs: 100,
      batchSize: 50,
      onEpochEnd: async (epoch, logs) => {
        console.info(`Epoch: ${epoch + 1}, Loss: ${logs.loss}`)
      },
    })
    .then(async () => {
      // Save the trained models
      await autoencoder.save(
        `file://${join(__dirname, 'weights', 'mnist', 'autoencoder')}`,
      )
      await encoder.save(
        `file://${join(__dirname, 'weights', 'mnist', 'encoder')}`,
      )
      await decoder.save(
        `file://${join(__dirname, 'weights', 'mnist', 'decoder')}`,
      )
    })
}

const fromWeights = async () => {
  const encoder = await tf.loadLayersModel(
    `file://${join(__dirname, 'weights', 'mnist', 'encoder', 'model.json')}`,
  )
  const decoder = await tf.loadLayersModel(
    `file://${join(__dirname, 'weights', 'mnist', 'decoder', 'model.json')}`,
  )

  const { image } = dataset.find((img) => img.label === 6)

  console.info(JSON.stringify(image))

  const xsTensor = tf.tensor2d([image])
  const compressed = encoder.predict(xsTensor)
  const decompressed = decoder.predict(compressed)

  // The output is thresholded during inference
  const decompressedImage = await decompressed.array()

  // Convert to 0, 1 based on threshold
  console.info(
    JSON.stringify(decompressedImage[0].map((pixel) => (pixel > 0.5 ? 1 : 0))),
  )
}

// Uncomment to train and save the model
fitAndSave()

// Uncomment to load and use the trained model
// fromWeights()
