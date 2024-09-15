const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_CLASSES = 10;
const BATCH_SIZE = 64;
const EPOCHS = 10;

async function loadMnist() {
  // Cargar los datos de MNIST directamente desde TensorFlow.js
  const [
    trainImagesResponse,
    trainLabelsResponse,
    testImagesResponse,
    testLabelsResponse,
  ] = await Promise.all([
    fetch(
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png"
    ),
    fetch(
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8"
    ),
    fetch(
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png"
    ),
    fetch(
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8"
    ),
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

function preprocessData(images, labels, numImages) {
  const xs = new Float32Array(numImages * IMAGE_WIDTH * IMAGE_HEIGHT);
  const ys = new Uint8Array(numImages);

  for (let i = 0; i < numImages; i++) {
    const offset = i * IMAGE_WIDTH * IMAGE_HEIGHT;
    for (let j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++) {
      xs[offset + j] = images[offset + j] / 255; // Asegura que las imágenes están correctamente normalizadas
    }
    ys[i] = labels[i];
  }

  return {
    xs: tf.tensor2d(xs, [numImages, IMAGE_WIDTH * IMAGE_HEIGHT]),
    ys: tf.oneHot(tf.tensor1d(ys, "int32"), NUM_CLASSES),
  };
}

// Crear el modelo ANN
function createModel() {
  const model = tf.sequential();

  // Capa de entrada
  model.add(
    tf.layers.dense({
      inputShape: [IMAGE_WIDTH * IMAGE_HEIGHT],
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), // Regularización L2
    })
  );

  // Primera capa oculta con 16 neuronas
  model.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );

  // Segunda capa oculta con 16 neuronas
  model.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );

  // Capa de salida con 10 neuronas para las clases (0 al 9)
  model.add(
    tf.layers.dense({
      units: NUM_CLASSES,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.0001), // Tasa de aprendizaje más baja
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function trainModel() {
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

  // Verifica los primeros datos de entrenamiento y etiquetas
  console.log(
    "First training image:",
    trainData.xs.slice([0, 0], [1, IMAGE_WIDTH * IMAGE_HEIGHT]).dataSync()
  );
  console.log(
    "First training label:",
    trainData.ys.slice([0, 0], [1, NUM_CLASSES]).dataSync()
  );

  console.log(trainData.xs.dataSync());

  // Verifica si hay NaN en los datos de entrenamiento
  if (trainData.xs.dataSync().some(isNaN)) {
    console.error("NaN found in training images!");
    return;
  }

  if (trainData.ys.dataSync().some(isNaN)) {
    console.error("NaN found in training labels!");
    return;
  }

  const model = createModel();

  // Entrenar el modelo
  await model.fit(trainData.xs, trainData.ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [testData.xs, testData.ys],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`
        );

        const newNode = document.createElement('div');
        newNode.innerHTML = `Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`;
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

// Llamar la función de entrenamiento al hacer clic en el botón
document.getElementById("train-btn").addEventListener("click", trainModel);
