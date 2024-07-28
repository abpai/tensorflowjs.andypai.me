import * as tf from '@tensorflow/tfjs'
import visualizeArchitecture from './js/network-visualization.js'

const fetchTestData = async () => {
  const response = await fetch(`${process.env.API_URL}/mnist?end=100&type=test`)
  const json = await response.json()
  return json
}

const drawImage = (image) => {
  const imageCanvas = document.createElement('canvas')

  const size = 14
  const scale = size * 8

  imageCanvas.width = scale
  imageCanvas.height = scale

  const ctx = imageCanvas.getContext('2d')
  const imageData = ctx.createImageData(size, size)
  image.forEach((pixel, idx) => {
    const value = pixel * 255
    imageData.data[idx * 4 + 0] = value
    imageData.data[idx * 4 + 1] = value
    imageData.data[idx * 4 + 2] = value
    imageData.data[idx * 4 + 3] = 255
  })

  // Create a temporary canvas to scale the image
  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = size
  tempCanvas.height = size
  const tempCtx = tempCanvas.getContext('2d')
  tempCtx.putImageData(imageData, 0, 0)

  // Scale up the image and draw it on the image canvas
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(tempCanvas, 0, 0, size, size, 0, 0, scale, scale)
  return imageCanvas
}

const main = async () => {
  const [test, autoencoder] = await Promise.all([
    fetchTestData(),
    tf.loadLayersModel(
      'https://storage.googleapis.com/andypai-me/tfjs/mnist-autoencoder/model.json',
    ),
  ])

  const table = visualizeArchitecture(autoencoder)
  const architecture = document.getElementById('architecture')
  architecture.innerHTML = '' // Clear previous content
  architecture.appendChild(table)

  for (const { image: original } of test) {
    const xsTensor = tf.tensor2d([original])
    const encoded = autoencoder.predict(xsTensor)
    const image = await encoded.array()
    const reconstructed = image[0].map((pixel) => (pixel > 0.5 ? 1 : 0))
    document.getElementById('original').innerHTML = ''
    document.getElementById('reconstructed').innerHTML = ''
    document.getElementById('original').appendChild(drawImage(original))
    document
      .getElementById('reconstructed')
      .appendChild(drawImage(reconstructed))
    await new Promise((resolve) => {
      setTimeout(resolve, 2000)
    })
  }
}

main()
