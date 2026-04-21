document.addEventListener("DOMContentLoaded", () => {
    // Note: If deploying, change this to your production backend URL
    const API_URL = "http://localhost:8000/api/analyze";

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPanel = document.getElementById("uploadPanel");
    const loader = document.getElementById("loader");
    const resultsPanel = document.getElementById("resultsPanel");
    const errorBanner = document.getElementById("errorBanner");
    const resetBtn = document.getElementById("resetBtn");

    // UI elements for Result
    const faceShapeVal = document.getElementById("faceShapeVal");
    const genderVal = document.getElementById("genderVal");
    const processedImage = document.getElementById("processedImage");
    const logicSummary = document.getElementById("logicSummary");
    const recommendationsOutput = document.getElementById("recommendationsOutput");

    // Click to upload
    dropZone.addEventListener("click", () => fileInput.click());

    // Drag and Drop Handling
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener("click", () => {
        resultsPanel.classList.add("hidden");
        errorBanner.classList.add("hidden");
        uploadPanel.classList.remove("hidden");
        fileInput.value = "";
    });

    function handleFile(file) {
        // Validate
        const validTypes = ["image/png", "image/jpeg", "image/jpg"];
        if (!validTypes.includes(file.type)) {
            showError("Invalid file type. Please upload a JPG or PNG.");
            return;
        }

        // Hide upload, show loader
        errorBanner.classList.add("hidden");
        uploadPanel.classList.add("hidden");
        loader.classList.remove("hidden");

        const formData = new FormData();
        formData.append("file", file);

        // Upload to Backend
        fetch(API_URL, {
            method: "POST",
            body: formData
        })
        .then(async (response) => {
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Server responded with an error.");
            }
            return data;
        })
        .then(data => {
            loader.classList.add("hidden");
            displayResults(data);
        })
        .catch(error => {
            loader.classList.add("hidden");
            uploadPanel.classList.remove("hidden");
            showError(error.message);
        });
    }

    function displayResults(data) {
        // Hydrate UI State
        faceShapeVal.textContent = data.shape;
        genderVal.textContent = data.gender;
        processedImage.src = `data:image/jpeg;base64,${data.image_base64}`;
        logicSummary.innerHTML = `<strong>Backend Math Logic:</strong><br/>${data.logic_summary}`;
        
        // Render Markdown
        // ensure marked is loaded from the CDN in index.html
        recommendationsOutput.innerHTML = marked.parse(data.recommendations);
        
        // Show Results
        resultsPanel.classList.remove("hidden");
    }

    function showError(msg) {
        errorBanner.textContent = msg;
        errorBanner.classList.remove("hidden");
    }
});
