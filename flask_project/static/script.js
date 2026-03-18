document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadText = document.getElementById('uploadText');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    const resultTitle = document.getElementById('resultTitle');
    const confidenceScore = document.getElementById('confidenceScore');

    let selectedFile = null;

    // --- 1. File Upload & Drag-and-Drop Logic ---

    // Click to open file dialog
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag over styling
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    // Drag leave styling
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle file selection via input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Process the selected file
    function handleFile(file) {
        // Basic validation
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError("Please upload a valid image file (JPG, PNG).");
            return;
        }

        selectedFile = file;
        submitBtn.disabled = false; // Enable submit button

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadText.classList.add('hidden');
        };
        reader.readAsDataURL(file);

        // Hide any previous results
        resultSection.classList.add('hidden');
    }

    // --- 2. Form Submission & API Integration ---

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!selectedFile) return;

        // Update UI to show loading state
        submitBtn.disabled = true;
        loading.classList.remove('hidden');
        resultSection.classList.add('hidden');

        // Prepare data to send to Flask
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Send POST request to Flask backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            // Parse JSON response
            const data = await response.json();

            // Check if backend returned an error status
            if (!response.ok) {
                throw new Error(data.error || 'Something went wrong on the server.');
            }

            // Display success result
            showResult(data);

        } catch (error) {
            // Display error message
            showError(error.message);
        } finally {
            // Remove loading state
            loading.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    // --- 3. UI Update Helpers ---

    function showResult(data) {
        // Reset classes
        resultSection.className = '';
        
        if (data.status === 'fractured') {
            resultSection.classList.add('result-fractured');
            resultTitle.innerHTML = '🔴 ' + data.result;
        } else {
            resultSection.classList.add('result-normal');
            resultTitle.innerHTML = '🟢 ' + data.result;
        }

        confidenceScore.textContent = `Confidence Score: ${data.prediction.toFixed(2)}`;
    }

    function showError(message) {
        resultSection.className = 'result-error';
        resultTitle.innerHTML = '⚠️ Error';
        confidenceScore.textContent = message;
    }
});
