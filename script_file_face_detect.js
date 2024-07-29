document.getElementById('fileInput').addEventListener('change', handleFileSelect, false);

async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 320, 240);
        await runInference(canvas);
    };
}

async function runInference(canvas) {
    const session = await ort.InferenceSession.create('./version-RFB-320.onnx');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, 320, 240);

    // Preprocess the image data to match the input format of the face detection model
    const input = new Float32Array(1 * 3 * 240 * 320);
    for (let i = 0; i < 240; i++) {
        for (let j = 0; j < 320; j++) {
            const idx = (i * 320 + j) * 4;
            const r = imageData.data[idx] / 255.0;
            const g = imageData.data[idx + 1] / 255.0;
            const b = imageData.data[idx + 2] / 255.0;
            input[(0 * 240 + i) * 320 + j] = r;
            input[(1 * 240 + i) * 320 + j] = g;
            input[(2 * 240 + i) * 320 + j] = b;
        }
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, 240, 320]);
    const feeds = { input: tensor };
    const results = await session.run(feeds);

    const boxes = results.boxes.data;
    const scores = results.scores.data;

    // Apply Non-Maximum Suppression
    const filteredBoxes = nonMaxSuppression(boxes, scores, 0.7, 0.5);

    // Draw bounding boxes
    for (const box of filteredBoxes) {
        const [x1, y1, x2, y2, score] = box;

        ctx.beginPath();
        ctx.rect(x1 * 320, y1 * 240, (x2 - x1) * 320, (y2 - y1) * 240);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'red';
        ctx.stroke();
    }
}

function nonMaxSuppression(boxes, scores, scoreThreshold, iouThreshold) {
    let filteredBoxes = [];

    // Filter out boxes with low scores
    let candidates = [];
    for (let i = 0; i < scores.length; i += 2) {
        if (scores[i + 1] > scoreThreshold) {
            candidates.push([boxes[i * 2], boxes[i * 2 + 1], boxes[i * 2 + 2], boxes[i * 2 + 3], scores[i + 1]]);
        }
    }

    // Sort candidates by score in descending order
    candidates.sort((a, b) => b[4] - a[4]);

    // Perform Non-Maximum Suppression
    while (candidates.length > 0) {
        const [x1, y1, x2, y2, score] = candidates.shift();
        filteredBoxes.push([x1, y1, x2, y2, score]);

        candidates = candidates.filter(box => {
            const iou = intersectionOverUnion([x1, y1, x2, y2], [box[0], box[1], box[2], box[3]]);
            return iou < iouThreshold;
        });
    }

    return filteredBoxes;
}

function intersectionOverUnion(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    const iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
}
