/* ═══════════════════════════════════════════════════════════
   GradeAI — Frontend Logic
═══════════════════════════════════════════════════════════ */

// ── SVG gradient (injected into DOM for the ring fill) ───────────────────────
document.body.insertAdjacentHTML('beforeend', `
  <svg class="ring-defs" aria-hidden="true">
    <defs>
      <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%"   stop-color="#7c5cfc"/>
        <stop offset="100%" stop-color="#38bdf8"/>
      </linearGradient>
    </defs>
  </svg>
`);

// ── Element refs ──────────────────────────────────────────────────────────────
const statusDot     = document.getElementById('status-dot');
const statusLabel   = document.getElementById('status-label');
const notReadyBanner= document.getElementById('not-ready-banner');

const form          = document.getElementById('grade-form');
const gradeBtn      = document.getElementById('grade-btn');
const gradeBtnText  = gradeBtn.querySelector('.grade-btn-text');
const gradeBtnIcon  = gradeBtn.querySelector('.grade-btn-icon');
const btnSpinner    = document.getElementById('btn-spinner');

const questionInput = document.getElementById('question-input');
const referenceInput= document.getElementById('reference-input');
const maxMarksInput = document.getElementById('max-marks-input');

const fileInput     = document.getElementById('file-input');
const uploadZone    = document.getElementById('upload-zone');
const uploadInner   = document.getElementById('upload-inner');
const previewImg    = document.getElementById('preview-img');
const fileNameLabel = document.getElementById('file-name-label');
const removeImgBtn  = document.getElementById('remove-img-btn');
const cameraBtn     = document.getElementById('camera-btn');
const cameraStrip   = document.getElementById('camera-strip');
const videoFeed     = document.getElementById('video-feed');
const captureBtn    = document.getElementById('capture-btn');
const snapCanvas    = document.getElementById('snap-canvas');

const studentTextInput = document.getElementById('student-text-input');
const imageModeFields  = document.getElementById('image-mode-fields');
const textModeFields   = document.getElementById('text-mode-fields');

const mainCard      = document.getElementById('main-card');
const resultsPanel  = document.getElementById('results-panel');
const backBtn       = document.getElementById('back-btn');

const scoreNumber   = document.getElementById('score-number');
const scoreDenom    = document.getElementById('score-denom');
const scorePct      = document.getElementById('score-pct');
const ringFill      = document.getElementById('ring-fill');
const scoreBarFill  = document.getElementById('score-bar-fill');
const barScoreLabel = document.getElementById('bar-score-label');
const gradeChip     = document.getElementById('grade-chip');
const ocrContent    = document.getElementById('ocr-content');

const modeBtns      = document.querySelectorAll('.mode-btn');
const modeIndicator = document.querySelector('.mode-indicator');
const toast         = document.getElementById('toast');

// ── State ─────────────────────────────────────────────────────────────────────
let currentMode       = 'image';   // 'image' | 'text'
let capturedImageBlob = null;      // from camera
let cameraStream      = null;
let bertReady         = false;

// ── Ring circumference = 2π × 68 ≈ 427.26 ───────────────────────────────────
const RING_CIRCUMFERENCE = 2 * Math.PI * 68;

// ─────────────────────────────────────────────────────────────────────────────
// Status Check
// ─────────────────────────────────────────────────────────────────────────────
async function checkStatus() {
    try {
        const res  = await fetch('/api/status');
        const data = await res.json();
        bertReady  = data.bert_ready;

        if (bertReady) {
            statusDot.className   = 'status-dot ready';
            statusLabel.textContent = 'BERT Model Ready';
            notReadyBanner.style.display = 'none';
        } else {
            statusDot.className   = 'status-dot not-ready';
            statusLabel.textContent = 'Model Not Trained';
            notReadyBanner.style.display = 'flex';
        }
    } catch {
        statusDot.className   = 'status-dot error';
        statusLabel.textContent = 'Server Offline';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode Switcher
// ─────────────────────────────────────────────────────────────────────────────
modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        if (mode === currentMode) return;
        currentMode = mode;

        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        if (mode === 'text') {
            modeIndicator.classList.add('right');
            imageModeFields.style.display = 'none';
            textModeFields.style.display  = 'block';
        } else {
            modeIndicator.classList.remove('right');
            imageModeFields.style.display = 'block';
            textModeFields.style.display  = 'none';
        }
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// File Upload / Drag-and-Drop
// ─────────────────────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', e => {
    if (e.target !== previewImg) fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});

function handleFile(file) {
    capturedImageBlob = null;   // reset camera capture
    fileNameLabel.textContent = file.name;
    const url = URL.createObjectURL(file);
    showPreview(url);
}

function showPreview(url) {
    previewImg.src = url;
    previewImg.style.display = 'block';
    uploadInner.style.display = 'none';
    uploadZone.classList.add('has-image');
    removeImgBtn.style.display = 'inline-flex';
}

removeImgBtn.addEventListener('click', () => {
    fileInput.value = '';
    capturedImageBlob = null;
    fileNameLabel.textContent = '';
    previewImg.src = '';
    previewImg.style.display = 'none';
    uploadInner.style.display = 'flex';
    uploadZone.classList.remove('has-image');
    removeImgBtn.style.display = 'none';
});

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────
cameraBtn.addEventListener('click', async () => {
    if (cameraStream) {
        stopCamera();
        return;
    }
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        videoFeed.srcObject = cameraStream;
        cameraStrip.style.display = 'flex';
        cameraBtn.innerHTML = `
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg> Close Camera`;
    } catch {
        showToast('Camera access denied or not available.');
    }
});

captureBtn.addEventListener('click', () => {
    snapCanvas.width  = videoFeed.videoWidth;
    snapCanvas.height = videoFeed.videoHeight;
    snapCanvas.getContext('2d').drawImage(videoFeed, 0, 0);
    snapCanvas.toBlob(blob => {
        capturedImageBlob = blob;
        fileNameLabel.textContent = 'camera_capture.jpg';
        showPreview(URL.createObjectURL(blob));
        stopCamera();
    }, 'image/jpeg', 0.92);
});

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    cameraStrip.style.display = 'none';
    cameraBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>
      </svg> Use Camera`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Form Submission
// ─────────────────────────────────────────────────────────────────────────────
form.addEventListener('submit', async e => {
    e.preventDefault();
    if (!bertReady) {
        showToast('⚠️ BERT model not ready. Train it first.');
        return;
    }

    setLoading(true);

    try {
        let result;

        if (currentMode === 'image') {
            // Get the file to send
            const file = capturedImageBlob
                ? new File([capturedImageBlob], 'capture.jpg', { type: 'image/jpeg' })
                : fileInput.files[0];

            if (!file) { showToast('Please upload or capture an image.'); setLoading(false); return; }

            const fd = new FormData();
            fd.append('image',            file);
            fd.append('question',         questionInput.value.trim());
            fd.append('reference_answer', referenceInput.value.trim());
            fd.append('max_marks',        maxMarksInput.value);
            fd.append('ocr_engine',       'trocr');

            const res = await fetch('/api/grade/bert', { method: 'POST', body: fd });
            result = await res.json();

        } else {
            // Text mode
            const fd = new FormData();
            fd.append('student_answer',   studentTextInput.value.trim());
            fd.append('question',         questionInput.value.trim());
            fd.append('reference_answer', referenceInput.value.trim());
            fd.append('max_marks',        maxMarksInput.value);

            const res = await fetch('/api/grade/bert/text', { method: 'POST', body: fd });
            result = await res.json();
        }

        if (!result.success) throw new Error(result.error || 'Grading failed.');

        showResults(result);

    } catch (err) {
        showToast('Error: ' + err.message);
    } finally {
        setLoading(false);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Results Display
// ─────────────────────────────────────────────────────────────────────────────
function showResults(data) {
    const score    = data.score;
    const maxMarks = data.max_marks;
    const pct      = data.percentage;   // 0–100

    // Hide form, show results
    mainCard.style.display     = 'none';
    resultsPanel.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Score text
    scoreNumber.textContent = score.toFixed(1);
    scoreDenom.textContent  = `/ ${maxMarks}`;
    scorePct.textContent    = `${pct}%`;

    // Animate ring (stroke-dashoffset: full → remaining)
    const offset = RING_CIRCUMFERENCE * (1 - pct / 100);
    ringFill.style.strokeDashoffset = offset;

    // Animate bar
    scoreBarFill.style.width = `${pct}%`;
    barScoreLabel.textContent = `${score.toFixed(1)} / ${maxMarks}`;

    // Grade chip
    const { label, cls } = gradeFor(pct);
    gradeChip.textContent = label;
    gradeChip.className   = `grade-chip grade-${cls}`;

    // OCR text
    ocrContent.textContent = data.extracted_text
        ? data.extracted_text
        : '(No text — manual answer graded directly)';
}

function gradeFor(pct) {
    if (pct >= 90) return { label: '🏆 Excellent',   cls: 'excellent' };
    if (pct >= 75) return { label: '✅ Good',         cls: 'good' };
    if (pct >= 55) return { label: '📘 Satisfactory', cls: 'satisfactory' };
    if (pct >= 35) return { label: '⚠️ Needs Work',   cls: 'needs-work' };
    return              { label: '❌ Insufficient',   cls: 'insufficient' };
}

// Inject grade chip styles
const gradeStyles = document.createElement('style');
gradeStyles.textContent = `
.grade-excellent    { background:rgba(34,211,165,.12); border:1px solid rgba(34,211,165,.3); color:#22d3a5; }
.grade-good         { background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.3); color:#38bdf8; }
.grade-satisfactory { background:rgba(124,92,252,.12); border:1px solid rgba(124,92,252,.3); color:#a78bfa; }
.grade-needs-work   { background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.3); color:#f59e0b; }
.grade-insufficient { background:rgba(244,63,94,.12);  border:1px solid rgba(244,63,94,.3);  color:#f43f5e; }
`;
document.head.appendChild(gradeStyles);

// ─────────────────────────────────────────────────────────────────────────────
// Back Button
// ─────────────────────────────────────────────────────────────────────────────
backBtn.addEventListener('click', () => {
    resultsPanel.style.display = 'none';
    mainCard.style.display     = 'block';

    // Reset ring & bar for next use
    ringFill.style.strokeDashoffset = RING_CIRCUMFERENCE;
    scoreBarFill.style.width        = '0%';
});

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function setLoading(on) {
    gradeBtn.disabled         = on;
    gradeBtnText.style.display = on ? 'none' : 'inline';
    gradeBtnIcon.style.display = on ? 'none' : 'flex';
    btnSpinner.style.display   = on ? 'block' : 'none';
}

let toastTimer = null;
function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────
checkStatus();
