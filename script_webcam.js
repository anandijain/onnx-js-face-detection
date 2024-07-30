async function setupCamera() {
    const video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function runInference(canvas, overlay, session) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const overlayCtx = overlay.getContext('2d');
    ctx.drawImage(document.getElementById('video'), 0, 0, 320, 240);
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

    console.log('Raw Boxes:', boxes);
    console.log('Raw Scores:', scores);
    console.log('Boxes length:', boxes.length);
    console.log('Scores length:', scores.length);

    // Apply Non-Maximum Suppression
    const filteredBoxes = nonMaxSuppression(boxes, scores, 0.5, 0.3);

    console.log('Filtered Boxes:', filteredBoxes);

    // Clear the previous drawings
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw bounding boxes
    for (const box of filteredBoxes) {
        const [x1, y1, x2, y2, score] = box;

        console.log(`Drawing box at (${x1 * 320}, ${y1 * 240}, ${(x2 - x1) * 320}, ${(y2 - y1) * 240}) with score ${score}`);

        overlayCtx.beginPath();
        overlayCtx.rect(x1 * 320, y1 * 240, (x2 - x1) * 320, (y2 - y1) * 240);
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeStyle = 'red';
        overlayCtx.stroke();
    }

    // Call inference on the next animation frame
    requestAnimationFrame(() => runInference(canvas, overlay, session));
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
    const yB = Math.min(boxB[3], boxB[3]);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    const iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
}

(async () => {
    const video = await setupCamera();
    video.play();

    const canvas = document.getElementById('canvas');
    const overlay = document.getElementById('overlay');
    const session = await ort.InferenceSession.create('./version-RFB-320.onnx');
    runInference(canvas, overlay, session);
})();
