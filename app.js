let model;

async function loadModel() {
    model = await tf.loadLayersModel('https://teachablemachine.withgoogle.com/models/IlZmvBc2q/model.json');
    console.log('Model loaded.');
}

async function predictFromUpload(imageElement) {
    let tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])  // Make sure this size matches the input size of your model
        .toFloat()
        .expandDims(0);

    const prediction = await model.predict(tensor);
    displayResults(prediction);
    tensor.dispose();
}

function displayResults(prediction) {
    const predictions = Array.from(prediction.dataSync());  // Convert prediction to array
    const labels = ['Leaves', 'Balls', 'Architecture', 'Baked Products', 'Clothing Items'];  // Your model's labels
    const resultsElement = document.getElementById('prediction');
    resultsElement.innerHTML = "";  // Clear previous results

    predictions.forEach((prob, index) => {
        const probPercentage = (prob * 100).toFixed(2);  // Convert to percentage
        resultsElement.innerHTML += `${labels[index]}: ${probPercentage}% <br>`;
    });
}

document.getElementById('upload').addEventListener('change', event => {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById('uploadedImage');
        img.src = e.target.result;
        img.onload = () => predictFromUpload(img); // Ensure the image is loaded before prediction
    };
    reader.readAsDataURL(event.target.files[0]);
});

window.onload = loadModel;
