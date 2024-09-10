const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_CLASSES = 10;

const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 65000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const IMAGE_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const LABELS_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

let datasetImages, datasetLabels;
let model;

// Cargar las imágenes y etiquetas de MNIST
async function loadMNISTData() {
  const imgResponse = await fetch(IMAGE_URL);
  const imgBuffer = await imgResponse.arrayBuffer();
  const imgData = new Uint8Array(imgBuffer);

  const labelResponse = await fetch(LABELS_URL);
  const labelBuffer = await labelResponse.arrayBuffer();
  const labelData = new Uint8Array(labelBuffer);

  return { imgData, labelData };
}

// Preprocesar los datos
function preprocessImages(imgData) {
  const images = new Float32Array(
    NUM_DATASET_ELEMENTS * IMAGE_WIDTH * IMAGE_HEIGHT
  );
  for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
    for (let j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; j++) {
      images[i * IMAGE_WIDTH * IMAGE_HEIGHT + j] =
        imgData[i * IMAGE_WIDTH * IMAGE_HEIGHT + j] / 255;
    }
  }
  return images;
}

function preprocessLabels(labelData) {
  return tf
    .tensor2d(
      labelData.slice(0, NUM_DATASET_ELEMENTS),
      [NUM_DATASET_ELEMENTS, 1],
      "int32"
    )
    .oneHot(NUM_CLASSES);
}

async function loadAndPrepareData() {
  const { imgData, labelData } = await loadMNISTData();
  datasetImages = preprocessImages(imgData);
  datasetLabels = preprocessLabels(labelData);
}

// Crear el modelo
function createModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, 1],
      kernelSize: 3,
      filters: 16,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function trainModel() {
  model = createModel();

  const trainX = tf.tensor4d(datasetImages, [
    NUM_TRAIN_ELEMENTS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  const trainY = datasetLabels;

  await model.fit(trainX, trainY, {
    epochs: 5,
    batchSize: 64,
    validationSplit: 0.1,
    callbacks: tf.callbacks.earlyStopping({ monitor: "val_loss" }),
  });
}

async function predictDigit(imageData) {
  const inputTensor = tf.browser
    .fromPixels(imageData, 1)
    .resizeNearestNeighbor([IMAGE_WIDTH, IMAGE_HEIGHT])
    .toFloat()
    .div(tf.scalar(255))
    .expandDims(0);

  const prediction = model.predict(inputTensor);
  const predictedDigit = prediction.argMax(1).dataSync()[0];
  return predictedDigit;
}

// Función para renderizar el canvas y obtener el dibujo
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mousemove", drawOnCanvas);

function drawOnCanvas(event) {
  if (!drawing) return;

  ctx.fillStyle = "black";
  ctx.fillRect(event.offsetX, event.offsetY, 10, 10);
}

// Limpiar el canvas
document.getElementById("clearButton").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Realizar predicción
document.getElementById("predictButton").addEventListener("click", async () => {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const predictedDigit = await predictDigit(imageData);
  document.getElementById(
    "result"
  ).innerText = `Dígito reconocido: ${predictedDigit}`;
});

// Cargar los datos y entrenar el modelo al cargar la página
(async () => {
  await loadAndPrepareData();
  await trainModel();
  document.getElementById("result").innerText =
    "Modelo entrenado, puedes empezar a dibujar.";
})();
