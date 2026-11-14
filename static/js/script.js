
// =====================================
// PERFECTHUE - FRONTEND INTERACTIONS
// =====================================

document.addEventListener("DOMContentLoaded", function () {

    const uploadForm = document.getElementById("uploadForm");
    const imageInput = document.getElementById("imageUpload");
    const fileName = document.getElementById("fileName");
    const loadingSpinner = document.getElementById("loadingSpinner");
    const uploadArea = document.querySelector(".upload-area");

    // Display selected image filename
    if (imageInput && fileName) {

        imageInput.addEventListener("change", function () {

            if (this.files.length > 0) {

                const file = this.files[0];

                const allowedTypes = [
                    "image/jpeg",
                    "image/png"
                ];

                if (!allowedTypes.includes(file.type)) {
                    fileName.textContent = "Please select a JPG or PNG image.";
                    imageInput.value = "";
                    return;
                }

                fileName.textContent = "Selected: " + file.name;
            }
        });
    }

    // Drag and drop functionality
    if (uploadArea && imageInput) {

        uploadArea.addEventListener("dragover", function (event) {
            event.preventDefault();
            uploadArea.classList.add("drag-active");
        });

        uploadArea.addEventListener("dragleave", function () {
            uploadArea.classList.remove("drag-active");
        });

        uploadArea.addEventListener("drop", function (event) {

            event.preventDefault();
            uploadArea.classList.remove("drag-active");

            if (event.dataTransfer.files.length > 0) {

                imageInput.files = event.dataTransfer.files;

                imageInput.dispatchEvent(
                    new Event("change", { bubbles: true })
                );
            }
        });
    }

    // Show loading indicator when submitting
    if (uploadForm) {

        uploadForm.addEventListener("submit", function (event) {

            if (!imageInput || !imageInput.files.length) {
                event.preventDefault();
                alert("Please select an image first.");
                return;
            }

            if (loadingSpinner) {
                loadingSpinner.style.display = "flex";
            }

            const submitButton = uploadForm.querySelector("button[type='submit']");

            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = "Analysing...";
            }
        });
    }

});
// Apply detected skin colour
document.querySelectorAll(".color-swatch").forEach(function (swatch) {
    const color = swatch.dataset.color;

    if (color) {
        swatch.style.backgroundColor = color;
    }
});

// Apply recommended foundation shade RGB colours
document.querySelectorAll(".rgb-swatch").forEach(function (swatch) {
    const r = swatch.dataset.r;
    const g = swatch.dataset.g;
    const b = swatch.dataset.b;

    if (r !== undefined && g !== undefined && b !== undefined) {
        swatch.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
    }
});
