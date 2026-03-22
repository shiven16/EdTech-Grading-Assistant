const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const gradeBtn = document.getElementById('grade-btn');
const btnText = document.querySelector('.btn-text');
const loader = document.getElementById('loader');

const inputView = document.getElementById('input-view');
const resultView = document.getElementById('result-view');
const backBtn = document.getElementById('back-btn');

// Camera Elements
const cameraBtn = document.getElementById('camera-btn');
const cameraContainer = document.getElementById('camera-container');
const videoFeed = document.getElementById('video-feed');
const captureBtn = document.getElementById('capture-btn');

let currentFile = null;
let stream = null;

// File Upload Logic
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (file.type.startsWith('image/')) {
        currentFile = file;
        fileNameDisplay.textContent = "Selected: " + file.name;
        stopCamera(); // Stop camera if file is uploaded
    } else {
        alert('Please select an image file.');
    }
}

// Camera Logic
cameraBtn.addEventListener('click', async () => {
    if (stream) {
        stopCamera();
        return;
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        videoFeed.srcObject = stream;
        cameraContainer.style.display = 'block';
        fileNameDisplay.textContent = "Camera active...";
        currentFile = null; // Clear picked file
    } catch (err) {
        alert('Unable to access camera. Please allow permissions or use file upload.');
        console.error(err);
    }
});

captureBtn.addEventListener('click', () => {
    if (!stream) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    canvas.getContext('2d').drawImage(videoFeed, 0, 0);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        handleFile(file);
    }, 'image/jpeg', 0.9);
});

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraContainer.style.display = 'none';
}

// Grade Submission Logic
gradeBtn.addEventListener('click', async () => {
    const conceptsText = document.getElementById('concepts-input').value.trim();
    
    if (!currentFile) {
        alert("Please upload or capture an image response first.");
        return;
    }
    
    if (!conceptsText) {
        alert("Please provide the expected concepts.");
        return;
    }

    const conceptsArray = conceptsText.split('\n').map(c => c.trim()).filter(c => c.length > 0);

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('concepts', JSON.stringify(conceptsArray));

    btnText.textContent = "Processing...";
    loader.style.display = "block";
    gradeBtn.style.opacity = "0.7";
    gradeBtn.style.pointerEvents = "none";

    try {
        const response = await fetch('/api/grade', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            // Switch views
            inputView.style.display = 'none';
            resultView.style.display = 'block';
            stopCamera();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to the grading engine.');
    } finally {
        btnText.textContent = "Grade Response";
        loader.style.display = "none";
        gradeBtn.style.opacity = "1";
        gradeBtn.style.pointerEvents = "auto";
    }
});

// View Switching
backBtn.addEventListener('click', () => {
    resultView.style.display = 'none';
    inputView.style.display = 'block';
});

function displayResults(data) {
    document.getElementById('sim-score').textContent = data.result.similarity_score.toFixed(2);
    document.getElementById('key-score').textContent = data.result.keyword_score.toFixed(2);
    document.getElementById('final-score').textContent = data.result.final_score.toFixed(2);

    document.getElementById('ocr-text').textContent = data.extracted_text || "(No text extracted)";

    const matchedList = document.getElementById('matched-list');
    const missingList = document.getElementById('missing-list');
    
    matchedList.innerHTML = '';
    missingList.innerHTML = '';

    const matchedConcepts = data.result.matched_concepts || [];
    const allConcepts = data.concepts || [];

    matchedConcepts.forEach(c => {
        const li = document.createElement('li');
        li.textContent = c;
        matchedList.appendChild(li);
    });

    allConcepts.forEach(c => {
        if (!matchedConcepts.includes(c)) {
            const li = document.createElement('li');
            li.textContent = c;
            missingList.appendChild(li);
        }
    });

    if (matchedConcepts.length === 0) {
        matchedList.innerHTML = '<li style="color:#64748b; font-style:italic;" class="empty">None matched</li>';
        const emptyLi = matchedList.querySelector('.empty');
        emptyLi.style.setProperty('padding-left', '0', 'important');
        emptyLi.style.setProperty('list-style', 'none', 'important');
    }
}
