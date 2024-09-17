const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_CLASSES = 10;
const BATCH_SIZE = 64;
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

const NUM_DATASET_ELEMENTS = 65000;

const IMAGE_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const LABELS_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

let model;
let numTrainImages;

async function loadMnist() {
  // Load MNIST data directly from TensorFlow.js

  // MNIST image and label data are downloaded as Uint8Array arrays from the TensorFlow URL.
  // Images are preprocessed by dividing them by 255 to normalize them between 0 and 1,
  // and labels are converted to one-hot format.

  const [
    trainImagesResponse,
    trainLabelsResponse,
    testImagesResponse,
    testLabelsResponse,
  ] = await Promise.all([
    fetch(IMAGE_URL),
    fetch(LABELS_URL),
    fetch(IMAGE_URL),
    fetch(LABELS_URL),
  ]);

  const trainImagesBuffer = await trainImagesResponse.arrayBuffer();
  const trainLabelsBuffer = await trainLabelsResponse.arrayBuffer();
  const testImagesBuffer = await testImagesResponse.arrayBuffer();
  const testLabelsBuffer = await testLabelsResponse.arrayBuffer();

  const trainImages = new Uint8Array(trainImagesBuffer);
  const trainLabels = new Uint8Array(trainLabelsBuffer);
  const testImages = new Uint8Array(testImagesBuffer);
  const testLabels = new Uint8Array(testLabelsBuffer);

  return {
    train: { images: trainImages, labels: trainLabels },
    test: { images: testImages, labels: testLabels },
  };
}

// Images are stored in a Float32Array and normalized.
// They are then converted to 2D tensors (tf.tensor2d) so that the model can process them.
// Labels are converted to one-hot format using tf.oneHot to be used as output by the model.

function preprocessData(imgData, labelData) {
  const images = new Float32Array(numTrainImages * IMAGE_SIZE);
  const labels = new Uint8Array(numTrainImages);

  for (let i = 0; i < numTrainImages; i++) {
    const offset = i * IMAGE_SIZE;
    for (let j = 0; j < IMAGE_SIZE; j++) {
      const newValue = imgData[offset + j] / 255;

      if (isNaN(newValue)) {
        images[offset + j] = 0;
      } else {
        images[offset + j] = newValue;
      }
    }
    labels[i] = labelData[i];
  }

  return {
    images: tf.tensor2d(images, [numTrainImages, IMAGE_SIZE]),
    labels: tf.oneHot(tf.tensor1d(Array.from(labels), "int32"), NUM_CLASSES),
  };
}

// Create the ANN model
function createModel() {
  const model = tf.sequential();

  // Get values from the form
  const hiddenLayers = parseInt(document.getElementById("hiddenLayers").value);
  const neuronsByLayer = parseInt(
    document.getElementById("neuronsByLayer").value
  );

  const activationFunction = document.getElementById("fn-activation").value;

  // Input layer
  model.add(
    tf.layers.dense({
      inputShape: [IMAGE_SIZE],
      units: 128,
      activation: activationFunction,
    })
  );

  // Create hidden layers based on the number
  for (let index = 0; index < hiddenLayers; index++) {
    model.add(
      tf.layers.dense({
        units: neuronsByLayer,
        activation: activationFunction,
      })
    );
  }

  // Create output layer for all the classes (0 to 9)
  model.add(
    tf.layers.dense({
      units: NUM_CLASSES,
      activation: "softmax",
    })
  );

  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// Train model
async function trainModel() {
  cleanResults();

  numTrainImages = parseInt(document.getElementById("numTrainImages").value);

  const { train, test } = await loadMnist();

  const dataSet = preprocessData(train.images, train.labels);
  const dataTestSet = preprocessData(test.images, test.labels);

  const trainImages = dataSet.images;
  const trainLabels = dataSet.labels;

  const testImages = dataTestSet.images;
  const testLabels = dataTestSet.labels;

  // Create the model
  model = createModel();

  const quantityEpochs = parseInt(
    document.getElementById("quantityEpochs").value
  );

  // Train the model
  await model.fit(trainImages, trainLabels, {
    epochs: quantityEpochs,
    batchSize: BATCH_SIZE,
    // validationData: [testImages, testLabels],
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const newNode = document.createElement("div");
        newNode.innerHTML = `Epoch ${epoch + 1}: Loss = ${
          logs.loss
        } - Accuracy = ${logs.acc}`;
        document.getElementById("metrics").appendChild(newNode);

        if (isNaN(logs.loss)) {
          document.getElementById("result").innerText =
            "NaN detected in loss. Stopping training.";
          model.stopTraining = true;
        }
      },
    },
  });

  document.getElementById("result").innerText = "Modelo entrenado con éxito.";
  document.getElementById("train-btn").disabled = false;
}

function cleanResults() {
  document.getElementById("result").innerHTML = "";
  document.getElementById("metrics").innerHTML = "";

  document.getElementById("result").innerText = "Entrenando modelo...";
  document.getElementById("train-btn").disabled = true;
}

// Click event for train model button
document.getElementById("train-btn").addEventListener("click", trainModel);
