const TARGET_WIDTH = 640;
const TARGET_HEIGHT = 640;

let model;
let selectedImageElement;
let selectedImageElementHeight;
let selectedImageElementWidth;
let selectedCanvas;

window.onload = async () => {
    model = await loadModel();

    displayCroppedImage(document.getElementById('snowboard'));
    displayCroppedImage(document.getElementById('room'));
    displayCroppedImage(document.getElementById('beach'));
    displayCroppedImage(document.getElementById('baseball'));
};


function selectImage(imgElement) {
    selectedImageElement = imgElement;
}

document.getElementById('file-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.src = e.target.result;
        img.onload = () => selectedImageElement = img;
    };
    reader.readAsDataURL(file);
});

async function imageToTensor(imageElement) {
    const canvas = document.getElementById('canvas');
    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    selectedImageElementHeight = imageElement.height;
    selectedImageElementWidth = imageElement.width;

    const scaleFactor = Math.min(TARGET_WIDTH / imageElement.width, TARGET_HEIGHT / imageElement.height);
    const scaledWidth = imageElement.width * scaleFactor;
    const scaledHeight = imageElement.height * scaleFactor;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    ctx.drawImage(imageElement, (TARGET_WIDTH - scaledWidth) / 2, (TARGET_HEIGHT - scaledHeight) / 2, scaledWidth, scaledHeight);

    const tensor = tf.browser.fromPixels(canvas);
    return tf.cast(tensor, 'float32').div(tf.scalar(255)).expandDims(0);
}

function imageToBase64(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    // Convert the canvas content to Base64
    return canvas.toDataURL();
}

async function loadClassNames(path) {
    try {
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching JSON:', error);
        throw error;
    }
}

async function processImage() {
    if (!selectedImageElement || !model) {
        console.error("No image selected or model not loaded");
        return;
    }
    const submitButton = document.getElementById('submit-button');
    submitButton.textContent = 'Processing'

    try {
        const inferenceImage = selectedImageElement;
        const tensor = await imageToTensor(inferenceImage);

        // Measure inference time
        const startTime = performance.now();
        const predictions = await runModel(model, tensor);
        const endTime = performance.now();
        const inferenceTime = endTime - startTime;
        document.getElementById('wasm-browser-inference-time').innerText = `Inference Time: ${inferenceTime.toFixed(2)} ms`;

        const classNames = await loadClassNames('./yolov8x_web_model/classes.json');
        const detections = processPredictions(predictions, classNames);

        await drawBoundingBoxes(inferenceImage, detections);
        await processAdditionalBackends(inferenceImage);
    } catch (e) {
        console.log(`Error: ${e}`)
        submitButton.textContent = 'Error'
    }
    submitButton.textContent = 'Submit'
}


async function processAdditionalBackends(imageElement) {
    const cpuEndpoint = document.getElementById('cpu-model-endpoint').value;
    const cpuResponseTime = document.getElementById('cpu-response-time');
    const cpuInferenceTime = document.getElementById('cpu-inference-time');

    const wasmEndpoint = document.getElementById('wasm-cpu-model-endpoint').value;
    const wasmResponseTime = document.getElementById('wasm-cpu-response-time');
    const wasmInferenceTime = document.getElementById('wasm-cpu-inference-time');
    
    try {
        let base64Image;
        if (cpuEndpoint || wasmEndpoint) {
            base64Image = imageToBase64(imageElement);
        }
        if (cpuEndpoint) {
            const startTime = performance.now();
            const response = await fetch(cpuEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });
            const endTime = performance.now();
            const responseData = await response.json()
            const responseTime = endTime - startTime;
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            console.log(`Response body: ${JSON.stringify(responseData)}`)
            cpuResponseTime.innerText = `Request Time: ${responseTime.toFixed(2)} ms`
            cpuInferenceTime.innerText = `Inference Time: ${responseData.inferenceTime} ms`;
        }
        if (wasmEndpoint) {
            const startTime = performance.now();
            const response = await fetch(wasmEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });
            const endTime = performance.now();
            const responseData = await response.json()
            const responseTime = endTime - startTime;
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            wasmResponseTime.innerText = `Request Time: ${responseTime.toFixed(2)} ms`
            wasmInferenceTime.innerText = `Inference Time: ${responseData.inferenceTime} ms`;
        }
    } catch (error) {
        console.error('Error fetching model:', error);
    }
}

function processPredictions(predictions, classNames) {
    return tf.tidy(() => {
        const transRes = predictions.transpose([0, 2, 1]);
        const boxes = calculateBoundingBoxes(transRes);
        const [scores, labels] = calculateScoresAndLabels(transRes, classNames);

        const indices = tf.image.nonMaxSuppression(boxes, scores, predictions.shape[2], 0.6, 0.45).arraySync();
        return extractSelectedPredictions(indices, boxes, labels, classNames);
    });
}

function calculateBoundingBoxes(transRes) {
    const [xCenter, yCenter, width, height] = [
        transRes.slice([0, 0, 0], [-1, -1, 1]),
        transRes.slice([0, 0, 1], [-1, -1, 1]),
        transRes.slice([0, 0, 2], [-1, -1, 1]),
        transRes.slice([0, 0, 3], [-1, -1, 1])
    ];

    const topLeftX = tf.sub(xCenter, tf.div(width, 2));
    const topLeftY = tf.sub(yCenter, tf.div(height, 2));
    return tf.concat([topLeftX, topLeftY, width, height], 2).squeeze();
}

function calculateScoresAndLabels(transRes, classNames) {
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, Object.keys(classNames).length]).squeeze(0);
    return [rawScores.max(1), rawScores.argMax(1)];
}

function extractSelectedPredictions(indices, boxes, labels, classNames) {
    return indices.map(i => {
        const box = boxes.slice([i, 0], [1, -1]).squeeze().arraySync();
        const label = labels.slice([i], [1]).arraySync()[0];
        return { box, label: classNames[label] };
    });
}

function displayCroppedImage(imageElement) {
    const displaySize = 256;

    const canvas = document.createElement('canvas');
    canvas.width = displaySize;
    canvas.height = displaySize;
    const ctx = canvas.getContext('2d');

    const cropSize = Math.min(imageElement.width, imageElement.height);
    const startX = (imageElement.width - cropSize) / 2;
    const startY = (imageElement.height - cropSize) / 2;

    ctx.drawImage(imageElement, startX, startY, cropSize, cropSize, 0, 0, displaySize, displaySize);

    canvas.onclick = () => {
        if (selectedCanvas) {
            selectedCanvas.classList.remove('canvas-selected');
        }
        canvas.classList.add('canvas-selected');
        selectedCanvas = canvas;
        selectImage(imageElement);
    };

    const container = document.getElementById('cropped-images-container');
    container.appendChild(canvas);
}


async function drawBoundingBoxes(imageElement, detections) {
    try {
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        canvas.width = imageElement.width;
        canvas.height = imageElement.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
        
        const resizeScale = Math.min(TARGET_WIDTH / canvas.width, TARGET_HEIGHT / canvas.height);
        const dx = (TARGET_WIDTH - canvas.width * resizeScale) / 2;
        const dy = (TARGET_HEIGHT - canvas.height * resizeScale) / 2;

        detections.forEach(({ box, label }) => {
            let [topLeftX, topLeftY, width, height] = box;
            topLeftX = topLeftX / resizeScale - dx / resizeScale;
            topLeftY = topLeftY / resizeScale - dy / resizeScale;
            width /= resizeScale;
            height /= resizeScale;

            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(topLeftX, topLeftY, width, height);
            ctx.fillStyle = 'red';
            ctx.font = '20px Arial';
            ctx.fillText(label, topLeftX, topLeftY - 7);
        });
    } catch (error) {
        console.error(`Error drawing bounding boxes: ${error.message}`);
        throw error;
    }
}

async function loadModel() {
    try {
        await tf.setBackend('wasm');
        return await tf.loadGraphModel('./yolov8x_web_model/model.json');
    } catch (error) {
        console.error(`Error loading model: ${error.message}`);
        throw error;
    }
}

async function runModel(model, tensor) {
    try {
        return model.predict(tensor);
    } catch (error) {
        console.error(`Error running model prediction: ${error.message}`);
        throw error;
    }
}
