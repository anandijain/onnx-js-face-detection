document.getElementById('fileInput').addEventListener('change', handleFileSelect, false);

async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 28, 28);
        runInference(canvas);
    };
}

async function runInference(canvas) {
    const session = await ort.InferenceSession.create('./mnist.onnx');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, 28, 28);

    // Preprocess the image data to match the input format of the MNIST model
    const input = new Float32Array(1 * 1 * 28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        const j = i * 4;
        input[i] = imageData.data[j] / 255.0;
    }

    const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
    const feeds = { Input3: tensor };
    const results = await session.run(feeds);
    const output = results.Plus214_Output_0.data;
    const prediction = output.indexOf(Math.max(...output));
    document.getElementById('result').innerText = `Prediction: ${prediction}`;
}
