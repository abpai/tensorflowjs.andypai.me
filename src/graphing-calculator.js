import { evaluate } from 'mathjs'
import * as tf from '@tensorflow/tfjs'
import GameEngine from './js/game-engine.js'
import range from './js/range.js'
import { generateTrainingData, buildModel } from './js/regression-model.js'
import visualizeArchitecture from './js/network-visualization.js'

const size = Math.min(+document.querySelector('.demo').offsetWidth, 600)

class GraphingCalculator extends GameEngine {
  constructor({
    canvas,
    lossCanvas,
    validationLossCanvas,
    modelArchitectureId,
    expression = 'sin(x)',
    xRange = { min: -5, max: 5 },
    yRange = { min: -5, max: 5 },
  }) {
    super({ canvas, width: size, height: size })
    this.expression = expression
    this.xRange = xRange
    this.yRange = yRange

    this.model = null
    this.trainId = null
    this.drawId = null
    this.maxEpochs = 100
    this.batchSize = 25
    this.trainingData = []
    this.validationData = []
    this.lossHistory = []
    this.validationLossHistory = []
    this.logs = {}

    this.lossCanvas = lossCanvas
    this.ctxLoss = this.lossCanvas.getContext('2d')
    this.lossCanvas.setAttribute('width', size)
    this.lossCanvas.setAttribute('height', 200)

    this.validationLossCanvas = validationLossCanvas
    this.ctxValidationLoss = this.validationLossCanvas.getContext('2d')
    this.validationLossCanvas.setAttribute('width', size)
    this.validationLossCanvas.setAttribute('height', 200)

    this.modelArchitectureId = modelArchitectureId
  }

  getExpression() {
    return this.expression
  }

  setExpression(newExpression) {
    this.expression = newExpression
  }

  mapX(x) {
    return (
      this.padding +
      ((x - this.xRange.min) / (this.xRange.max - this.xRange.min)) *
        this.innerWidth
    )
  }

  mapY(y) {
    return (
      this.height -
      this.padding -
      ((y - this.yRange.min) / (this.yRange.max - this.yRange.min)) *
        this.innerHeight
    )
  }

  displayModelArchitecture() {
    if (this.model && this.modelArchitectureId) {
      const table = visualizeArchitecture(this.model)
      this.modelArchitectureId.innerHTML = '' // Clear previous content
      this.modelArchitectureId.appendChild(table)
    }
  }

  plotFunction(fn, { color = '#f6511d', lineWidth = 1, lineDash = [] } = {}) {
    const expression = this.getExpression()
    if (!fn) fn = (x) => evaluate(expression, { x })

    this.ctx.lineWidth = lineWidth
    this.ctx.strokeStyle = color
    this.ctx.setLineDash(lineDash)
    this.ctx.beginPath()

    range(this.xRange.min, this.xRange.max + 1, 0.1).forEach((x, i) => {
      const xPos = this.mapX(x)
      const yPos = this.mapY(fn(x))

      if (i === 0) {
        this.ctx.moveTo(xPos, yPos)
      } else {
        this.ctx.lineTo(xPos, yPos)
      }
    })

    this.ctx.stroke()
  }

  plotCurve(ctx, lossData, title, color) {
    const { width, height } = ctx.canvas
    const padding = 25
    const plotWidth = width - 2 * padding
    const plotHeight = height - 2 * padding

    // Clear the previous plot
    ctx.clearRect(0, 0, width, height)

    // Find the maximum loss for scaling
    const maxLoss = Math.max(...lossData)

    // Plot loss curve
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()

    lossData.forEach((loss, index) => {
      const x = padding + (index / (this.totalBatches - 1)) * plotWidth
      const y = height - padding - (loss / maxLoss) * plotHeight

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()
    ctx.closePath()

    // Draw axes and labels
    this.drawAxesAndLabels(ctx, width, height, padding)
  }

  // eslint-disable-next-line class-methods-use-this
  drawAxesAndLabels(ctx, width, height, padding) {
    // Draw x-axis
    ctx.strokeStyle = '#100c09'
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.stroke()
    ctx.closePath()

    // Draw y-axis
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(padding, padding)
    ctx.stroke()
    ctx.closePath()

    // Add x-axis label
    ctx.fillStyle = '#100c09'
    ctx.font = '12px Arial'
    ctx.fillText('Epochs', width / 2, height - padding / 4)

    // Add y-axis label
    ctx.save()
    ctx.translate(padding / 2, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('Loss', 0, 0)
    ctx.restore()
  }

  plotLossCurve() {
    this.plotCurve(this.ctxLoss, this.lossHistory)
  }

  plotValidationLossCurve() {
    this.plotCurve(this.ctxValidationLoss, this.validationLossHistory)
  }

  setup() {
    const func = document.getElementById('fnInput')
    if (!func.value) func.value = this.getExpression()

    this.setExpression(func.value)

    console.info(`New model for expression: ${func.value}`)
    this.model = buildModel()

    this.displayModelArchitecture()

    this.trainingData = generateTrainingData(this.xRange, (x) =>
      evaluate(func.value, { x }),
    )
    this.validationData = generateTrainingData(
      this.xRange,
      (x) => evaluate(func.value, { x }),
      { n: 50 },
    )
    this.lossHistory = []
    this.validationLossHistory = []
    this.currentEpoch = 0
    this.currentBatch = 0

    // Convert training data to tensors
    const [xs, ys] = this.trainingData
    this.xsTensor = tf.tensor2d(xs)
    this.ysTensor = tf.tensor1d(ys)

    // Convert validation data to tensors
    const [valXs, valYs] = this.validationData
    this.valXsTensor = tf.tensor2d(valXs)
    this.valYsTensor = tf.tensor1d(valYs)

    // Calculate total number of batches
    this.totalBatches = Math.ceil(xs.length / this.batchSize) * this.maxEpochs

    // Start the training process
    this.train(this.batchSize)
  }

  calculateValidationLoss() {
    const predictions = this.model.predict(this.valXsTensor)
    const reshapedValYsTensor = this.valYsTensor.reshape([
      this.valYsTensor.shape[0],
      1,
    ])
    const loss = tf.losses.meanSquaredError(reshapedValYsTensor, predictions)
    return loss.dataSync()[0]
  }

  async train(batchSize) {
    await this.model.fit(this.xsTensor, this.ysTensor, {
      epochs: this.maxEpochs,
      shuffle: true,
      batchSize,
      callbacks: {
        onEpochBegin: (epoch) => {
          this.currentEpoch = epoch + 1
          this.currentBatch = 0
        },
        onBatchEnd: (batch, logs) => {
          this.currentBatch = batch + 1
          this.logs = {
            ...logs,
            lr: this.model.optimizer.learningRate,
          }
          this.lossHistory.push(logs.loss)
          this.validationLossHistory.push(this.calculateValidationLoss())
        },
        onEpochEnd: async (epoch, logs) => {
          // const valLoss = await this.calculateValidationLoss()
          // this.validationLossHistory.push(valLoss)
          // console.info(
          //   `Epoch ${epoch + 1}: loss = ${logs.loss}, val_loss = ${valLoss}`,
          // )
        },
        onTrainEnd: () => {
          console.info('Training finished')
          this.stop()
        },
      },
    })
  }

  draw() {
    this.clear()
    this.drawCoordinatePlan(
      this.xRange.min,
      this.xRange.max,
      this.yRange.min,
      this.yRange.max,
    )
    this.plotFunction((x) => evaluate(this.getExpression(), { x }), {
      color: '#f6511d',
      lineWidth: 3,
    })
    this.plotFunction(
      (x) => {
        const prediction = this.model.predict(tf.tensor2d([[x]])).dataSync()[0]
        return prediction
      },
      {
        color: '#006632',
        lineWidth: 3,
        lineDash: [10, 5],
      },
    )
    this.plotLossCurve()
    this.plotValidationLossCurve()

    const formatter = new Intl.NumberFormat('en-US', {
      style: 'decimal',
      maximumFractionDigits: 0,
    })

    const progress =
      ((this.currentEpoch - 1) * (this.totalBatches / this.maxEpochs) +
        this.currentBatch) /
      this.totalBatches
    document.querySelector('#epoch .value').innerText =
      `${formatter.format(this.currentEpoch)} (${(progress * 100).toFixed(1)}%)`

    document.querySelector('#loss .value').innerText = +this.logs.loss
      ? this.logs.loss.toFixed(5)
      : 'N/A'

    document.querySelector('#lr .value').innerText = +this.logs.lr
      ? this.logs.lr.toFixed(5)
      : 'N/A'
  }

  restart() {
    this.stop()
    this.start()
  }
}

const calculator = new GraphingCalculator({
  canvas: document.getElementById('plane'),
  lossCanvas: document.getElementById('lossCurve'),
  validationLossCanvas: document.getElementById('validationLossCurve'),
  modelArchitectureId: document.getElementById('architecture'),
})
calculator.start()

const reload = (e) => {
  if (e) e.preventDefault()

  calculator.setExpression(document.getElementById('fnInput').value)
  calculator.restart()
}

document.querySelector('.fn').addEventListener('submit', reload)
document.getElementById('plot').addEventListener('click', reload)
