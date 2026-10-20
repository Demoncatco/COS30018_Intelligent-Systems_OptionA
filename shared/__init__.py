"""Code shared between the model service and the agents. Keep this package dependency-light."""

"""Evaluation Agent planned architecutre - Justin
Reading reports using MedGemma with the intended output of 3* variables

- Abnormality                Lists the detected abnormality similar to how the original dataset's binary classification
- Severity                   Lists the severity of detected abnormality if possible (manual wording may be need to detect)
- Observation (TBD)          Possible inclusion of the medical report's reccomendations regarding detected abnormality or no further action

These variables will be attached to the study ID in a evaluation csv/ dataset for the reccomendation agent and coodinator agent to use

Further iterations should include the analysis of the DICOM scans included, using the compiled info from the model_service agent
These variables could include data such as:
- Abnormality detected 
- Confidence score behind detection
- Further evaluation/ confidence data for training/ testing use

This info would be additionally attached to the same evaluation csv/ dataset as the prior data or another file may be generated for seperation purposes
Further discussions required for DICOM analysis evaluations
"""