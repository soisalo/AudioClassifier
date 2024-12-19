# AudioClassifier

This project is an audio classification tool developed as part of the COMP:SGN.120 course project. The tool is capable of converting various audio file formats to .wav,
normalizing audio data, segmenting audio, extracting features, and classifying audio using a Support Vector Machine (SVM).

## Authors

- Eemil Soisalo
- Aaro Varjus

## Features

- **Audio Conversion**: Converts audio files to .wav format.
- **Audio Normalization**: Normalizes audio data.
- **Audio Segmentation**: Segments audio into fixed-length chunks, excluding silent segments.
- **Feature Extraction**: Extracts features such as MFCC, chroma, spectral contrast, and mel-spectrogram.
- **Classification**: Classifies audio using a Support Vector Machine (SVM).
- **Validation**: Evaluates the model on validation data.

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `librosa`
- `pydub`
- `scikit-learn`
- `matplotlib`

## Directory Structure

- Input directories for car and tram audio data:
  - `dir_input_car`
  - `dir_input_tram`
- Validation directories:
  - `dir_validation_input_tram`
  - `dir_validation_output_tram`
  - `dir_validation_output_car`
- Output directories for converted .wav files:
  - `dir_audio_output_car`
  - `dir_audio_output_tram`

## Usage

1. **Convert Audio Files to .wav**:
   ```python
   folder_convert(dir_input_car, dir_audio_output_car)
   folder_convert(dir_input_tram, dir_audio_output_tram)
   folder_convert(dir_validation_input_tram, dir_validation_output_tram)

Normalize Audio Data:

Python
normalized_car, sr = load_and_normalize_to_matrix(dir_audio_output_car)
normalized_tram, sr = load_and_normalize_to_matrix(dir_audio_output_tram)
normalized_car_validation, sr = load_and_normalize_to_matrix(dir_validation_output_car)
normalized_tram_validation, sr = load_and_normalize_to_matrix(dir_validation_output_tram)
Segment Audio Data:

Python
segment_car = segment_audio(normalized_car, sr, segment_length=3)
segment_tram = segment_audio(normalized_tram, sr, segment_length=3)
segment_car_validation = segment_audio(normalized_car_validation, sr, segment_length=3)
segment_tram_validation = segment_audio(normalized_tram_validation, sr, segment_length=3)
Extract Features:

Python
features_car = np.array([feature_extraction(segment, sr) for segment in segment_car])
features_tram = np.array([feature_extraction(segment, sr) for segment in segment_tram])
features_car_validation = np.array([feature_extraction(segment, sr) for segment in segment_car_validation])
features_tram_validation = np.array([feature_extraction(segment, sr) for segment in segment_tram_validation])
Train the Classifier:

Python
features = np.vstack([features_car, features_tram])
classes = np.concatenate([np.zeros(len(features_car)), np.ones(len(features_tram))])
results = classifier(features, classes)
Validate the Model:

Python
features_validation = np.vstack([features_car_validation, features_tram_validation])
classes_validation = np.concatenate([np.zeros(len(features_car_validation)), np.ones(len(features_tram_validation))])
validate_model(features_validation, classes_validation, results["model"], results["scaler"])


Running the Project

To run the project, simply execute the main() function:

    main()


License
This project is licensed under the MIT License.

You can add this to a `README.md` file in the root of your repository.
