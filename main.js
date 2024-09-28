import * as tf from "@tensorflow/tfjs";

import "./style.css";

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const MNIST_IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT; // 784
const NUM_CLASSES = 10; // 10 digits 0 to 9
const NUM_DATASET_ELEMENTS = 65000; // Total images
const BATCH_SIZE = 64;

let NUM_TRAIN_ELEMENTS = 55000; // 55000 default value
let NUM_TEST_ELEMENTS = 10000; // 10000 default value
let model; // Global model

const MNIST_IMAGES_SPRITE_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

// Load images and data from MnistData
async function loadMnistData() {
  const img = new Image();
  img.src = MNIST_IMAGES_SPRITE_PATH;
  img.crossOrigin = "Anonymous";
  await new Promise((resolve) => {
    img.onload = resolve;
  });

  // Create canvas to extract images
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const datasetBytesBuffer = new Float32Array(
    NUM_DATASET_ELEMENTS * MNIST_IMAGE_SIZE,
  );

  img.width = img.naturalWidth;
  img.height = img.naturalHeight;
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  // Extract every image of sprite
  for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
    const x = (i % 100) * IMAGE_WIDTH;
    const y = Math.floor(i / 100) * IMAGE_HEIGHT;
    const imageData = ctx.getImageData(x, y, IMAGE_WIDTH, IMAGE_HEIGHT);
    const pixels = imageData.data;
    for (let j = 0; j < MNIST_IMAGE_SIZE; j++) {
      datasetBytesBuffer[i * MNIST_IMAGE_SIZE + j] = pixels[j * 4] / 255; // Normalización
    }
  }

  // Load labels
  const labelsResponse = await fetch(MNIST_LABELS_PATH);
  const labelsArrayBuffer = await labelsResponse.arrayBuffer();
  const labels = new Uint8Array(labelsArrayBuffer);

  // Create tensor for images and labels
  const images = tf.tensor2d(datasetBytesBuffer, [
    NUM_DATASET_ELEMENTS,
    MNIST_IMAGE_SIZE,
  ]);
  const labelsTensor = tf.tensor2d(labels, [NUM_DATASET_ELEMENTS, NUM_CLASSES]);

  // Split training and test data
  const trainImages = images.slice(
    [0, 0],
    [NUM_TRAIN_ELEMENTS, MNIST_IMAGE_SIZE],
  );
  const testImages = images.slice(
    [NUM_TRAIN_ELEMENTS, 0],
    [NUM_TEST_ELEMENTS, MNIST_IMAGE_SIZE],
  );
  const trainLabels = labelsTensor.slice(
    [0, 0],
    [NUM_TRAIN_ELEMENTS, NUM_CLASSES],
  );
  const testLabels = labelsTensor.slice(
    [NUM_TRAIN_ELEMENTS, 0],
    [NUM_TEST_ELEMENTS, NUM_CLASSES],
  );

  return { trainImages, trainLabels, testImages, testLabels };
}

// Stop train model
function stopTrainModel() {
  model.stopTraining = true;
  document.getElementById('train-btn').disabled = false;
  document.getElementById('stop-btn').style.display = 'none';
}

// Create the ANN model
function createModel() {
  const model = tf.sequential();

  // Get values from the form
  const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
  const neuronsByLayer = parseInt(
    document.getElementById('neuronsByLayer').value
  );
  const activationFunction = document.getElementById('fn-activation').value;

  // Input layer
  model.add(
    tf.layers.dense({
      inputShape: [MNIST_IMAGE_SIZE],
      units: 28,
      activation: activationFunction
    })
  );

  // Create hidden layers based on the number
  for (let index = 0; index < hiddenLayers; index++) {
    model.add(
      tf.layers.dense({
        units: neuronsByLayer,
        activation: activationFunction
      })
    );
  }

  // Create output layer for all the classes (0 to 9)
  model.add(
    tf.layers.dense({
      units: NUM_CLASSES,
      activation: 'softmax'
    })
  );

  // Compile the model
  model.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// Create and train model
async function createAndTrainModel() {
  cleanResults();

  const porcentageTrainImages = parseInt(document.getElementById('numTrainImages').value);

  NUM_TRAIN_ELEMENTS = (NUM_DATASET_ELEMENTS * porcentageTrainImages) / 100;
  NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

  const { trainImages, trainLabels, testImages, testLabels } =
    await loadMnistData();

  // Create the model
  model = createModel();

  const quantityEpochs = parseInt(
    document.getElementById('quantityEpochs').value
  );

  // Train the model
  await model.fit(trainImages, trainLabels, {
    batchSize: BATCH_SIZE,
    epochs: quantityEpochs,
    validationData: [testImages, testLabels],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const newNode = document.createElement('div');
        newNode.innerHTML = `Epoch ${epoch + 1}: Loss = ${
          logs.loss
        } - Accuracy = ${logs.acc}`;
        document.getElementById('metrics').appendChild(newNode);

        if (isNaN(logs.loss)) {
          document.getElementById('result').innerText =
            'NaN detected in loss. Stopping training.';
          model.stopTraining = true;
        }
      }
    }
  });

  document.getElementById('result').innerText = 'Modelo entrenado con éxito.';
  document.getElementById('train-btn').disabled = false;
  document.getElementById('stop-btn').style.display = 'none';
}

function cleanResults() {
  document.getElementById('result').innerHTML = '';
  document.getElementById('metrics').innerHTML = '';

  document.getElementById('result').innerText = 'Entrenando modelo...';
  document.getElementById('train-btn').disabled = true;
  document.getElementById('stop-btn').style.display = 'block';
}

// Click event for train model button
document.getElementById('train-btn').addEventListener('click', createAndTrainModel);
document.getElementById('stop-btn').addEventListener('click', stopTrainModel);
