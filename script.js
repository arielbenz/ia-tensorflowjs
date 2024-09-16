const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_CLASSES = 10;
const BATCH_SIZE = 64;

const trainImagesUrl =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const trainLabelsUrl =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

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
    fetch(trainImagesUrl),
    fetch(trainLabelsUrl),
    fetch(trainImagesUrl),
    fetch(trainLabelsUrl),
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
function preprocessData(images, labels, numImages) {
  const xs = new Float32Array(numImages * IMAGE_WIDTH * IMAGE_HEIGHT);
  const ys = new Uint8Array(numImages);

  for (let i = 0; i < numImages; i++) {
    const offset = i * IMAGE_WIDTH * IMAGE_HEIGHT;
    for (let j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++) {
      xs[offset + j] = images[offset + j] / 255; // Normalize images
    }
    ys[i] = labels[i];
  }

  return {
    xs: tf.tensor2d(xs, [numImages, IMAGE_WIDTH * IMAGE_HEIGHT]),
    ys: tf.oneHot(tf.tensor1d(ys, "int32"), NUM_CLASSES),
  };
}

// Create the ANN model
function createModel(hiddenLayers, neuronsByLayer) {
  const model = tf.sequential();

  // Input layer
  model.add(
    tf.layers.dense({
      inputShape: [IMAGE_WIDTH * IMAGE_HEIGHT],
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), // Regularization L2
    })
  );

  // Create hidden layers based on the number
  for (let index = 0; index < hiddenLayers; index++) {
    model.add(
      tf.layers.dense({
        units: neuronsByLayer,
        activation: "relu",
        kernelInitializer: "heNormal",
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
    optimizer: tf.train.adam(0.0001), // Lower learning rate
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

function cleanResults() {
  document.getElementById("result").innerHTML = "";
  document.getElementById("metrics").innerHTML = "";
}

async function trainModel() {
  cleanResults();

  document.getElementById("result").innerText = "Entrenando modelo...";
  document.getElementById("train-btn").disabled = true;

  const mnist = await loadMnist();

  const numTrainImages = 1000;
  const numTestImages = 10000;

  const trainData = preprocessData(
    mnist.train.images,
    mnist.train.labels,
    numTrainImages
  );
  const testData = preprocessData(
    mnist.test.images,
    mnist.test.labels,
    numTestImages
  );

  // Verify labels and training data
  console.log(
    "First training image:",
    trainData.xs.slice([0, 0], [1, IMAGE_WIDTH * IMAGE_HEIGHT]).dataSync()
  );
  console.log(
    "First training label:",
    trainData.ys.slice([0, 0], [1, NUM_CLASSES]).dataSync()
  );

  // Verify if there is a NaN in training data
  if (trainData.xs.dataSync().some(isNaN)) {
    console.error("NaN found in training images!");
    return;
  }

  if (trainData.ys.dataSync().some(isNaN)) {
    console.error("NaN found in training labels!");
    return;
  }

  // Get values from the form
  const hiddenLayers = parseInt(document.getElementById("hiddenLayers").value);
  const neuronsByLayer = parseInt(
    document.getElementById("neuronsByLayer").value
  );
  const quantityEpochs = parseInt(
    document.getElementById("quantityEpochs").value
  );

  // Create the model
  const model = createModel(hiddenLayers, neuronsByLayer);

  // Train the model
  await model.fit(trainData.xs, trainData.ys, {
    epochs: quantityEpochs,
    batchSize: BATCH_SIZE,
    validationData: [testData.xs, testData.ys],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`
        );

        const newNode = document.createElement("div");
        newNode.innerHTML = `Epoch ${epoch + 1}: Loss = ${
          logs.loss
        }, Accuracy = ${logs.acc}`;
        document.getElementById("metrics").appendChild(newNode);

        if (isNaN(logs.loss)) {
          console.error("NaN detected in loss. Stopping training.");
          model.stopTraining = true;
        }
      },
    },
  });

  document.getElementById("result").innerText = "Modelo entrenado con éxito.";
  document.getElementById("train-btn").disabled = false;
}

// Click event for train model button
document.getElementById("train-btn").addEventListener("click", trainModel);
