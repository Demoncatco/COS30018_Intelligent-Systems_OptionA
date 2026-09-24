const submitBtn = document.getElementById("submitBtn");

const inputStatus = document.getElementById("inputStatus");
const diagnosticStatus = document.getElementById("diagnosticStatus");
const coordinatorStatus = document.getElementById("coordinatorStatus");
const recommendationStatus = document.getElementById("recommendationStatus");
const resultStatus = document.getElementById("resultStatus");

const result = document.getElementById("result");
const errorBox = document.getElementById("errorBox");
const errorMessage = document.getElementById("errorMessage");

const task = document.getElementById("task");
const file = document.getElementById("file");


function setStatus(element, type, text) {

    const dot = element.querySelector(".dot");

    dot.className = "dot " + type;

    element.lastChild.textContent = " " + text;
}


function resetStatus() {

    setStatus(inputStatus, "waiting", "Input Received");
    setStatus(diagnosticStatus, "waiting", "Diagnostic Agent");
    setStatus(coordinatorStatus, "waiting", "Coordinator Agent");
    setStatus(recommendationStatus, "waiting", "Recommendation Agent");
    setStatus(resultStatus, "waiting", "Result Generation");

    result.innerHTML = `
        <h3>System Output</h3>
        <p class="placeholder">
            Waiting for system results...
        </p>
    `;

    errorBox.style.display = "none";
}


submitBtn.addEventListener("click", function () {

    // Check if user entered a task or selected a file
    if (task.value.trim() === "" && file.files.length === 0) {

        errorBox.style.display = "block";

        errorMessage.textContent =
            "Please enter a task or upload an MRI study before submitting.";

        return;
    }

    errorBox.style.display = "none";

    submitBtn.disabled = true;
    submitBtn.textContent = "Processing...";

    resetStatus();

    // Input received
    setStatus(inputStatus, "completed", "Input Received");

    // Diagnostic Agent
    setTimeout(function () {

        setStatus(
            diagnosticStatus,
            "processing",
            "Diagnostic Agent - Processing"
        );

    }, 1000);


    setTimeout(function () {

        setStatus(
            diagnosticStatus,
            "completed",
            "Diagnostic Agent - Complete"
        );

    }, 3000);


    // Coordinator Agent
    setTimeout(function () {

        setStatus(
            coordinatorStatus,
            "processing",
            "Coordinator Agent - Checking"
        );

    }, 3000);


    setTimeout(function () {

        setStatus(
            coordinatorStatus,
            "completed",
            "Coordinator Agent - Complete"
        );

    }, 5000);


    // Recommendation Agent
    setTimeout(function () {

        setStatus(
            recommendationStatus,
            "processing",
            "Recommendation Agent - Processing"
        );

    }, 5000);


    setTimeout(function () {

        setStatus(
            recommendationStatus,
            "completed",
            "Recommendation Agent - Complete"
        );

    }, 7000);


    // Final result
    setTimeout(function () {

        setStatus(
            resultStatus,
            "processing",
            "Generating Result"
        );

    }, 7000);


    setTimeout(function () {

        setStatus(
            resultStatus,
            "completed",
            "Result Generation - Complete"
        );

        result.innerHTML = `
            <h3>System Output</h3>

            <p>
                <strong>Analysis completed successfully.</strong>
            </p>

            <p>
                The Diagnostic Agent processed the submitted input,
                the Coordinator Agent reviewed the output, and the
                Recommendation Agent generated the final response.
            </p>

            <p>
                <strong>Status:</strong> Analysis completed
            </p>
        `;

        submitBtn.disabled = false;
        submitBtn.textContent = "Submit Analysis";

    }, 8000);

});