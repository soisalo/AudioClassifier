"""
Eemil Soisalo 150353416
Aaro Varjus 

COMP:SGN.120 Project
Audio classification project

This project is an audio classification tool developed as part of the COMP:SGN.120 course project. 
The tool is capable of converting various audio file formats to .wav, normalizing audio data, segmenting audio,
extracting features, and classifying audio using a Support Vector Machine (SVM). In this case we studied the 
audiospectral difference of cars and trams.
"""


import os
import numpy as np
import librosa
import pydub

from pydub import AudioSegment
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#Directories for the .wav data
dir_input_car = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Auto/Audio_car_input"
dir_input_tram = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Tram/Audio_Tram_input"

#Validation directory
dir_validation_input_tram = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Tram/Audio_Tram_Validation_Source"
dir_validation_output_tram = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Tram/Audio_Tram_Validation"

#All car validation data was already converted to .wav
dir_validation_output_car = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Auto/Audio_car_Validation" 

#.wav conversion
dir_audio_output_car = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Auto/Audio_car_output"
dir_audio_output_tram = "C:/Users/eemil/Documents/COMP.SGN.120/Audio/Audio_Tram/Audio_tram_output"



def convert_to_wav(input_file, output_file):
    """
    Helper function to convert audio files to .wav
    """
    try:
        audio = AudioSegment.from_file(input_file)
    
        # Export as .wav
        audio.export(output_file, format="wav")
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"error converting {input_file}")

def folder_convert(dir_input, dir_output):
    """
    converts the whole folder to wav.
    """
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    input_files = [
        file_name for file_name in os.listdir(dir_input)
        if file_name.lower().endswith(("mp3", "ogg", "aac", "flac", "m4a"))
    ]

    for file_name in input_files:
            input_file = os.path.join(dir_input, file_name)
            output_file = os.path.join(dir_output, os.path.splitext(file_name)[0] + ".wav")
            convert_to_wav(input_file, output_file)

    # Check if all files have been converted
    converted_files = [
        os.path.splitext(file_name)[0] + ".wav" for file_name in input_files
    ]
    output_files = os.listdir(dir_output)

    if all(file in output_files for file in converted_files):
        print("All files have been successfully converted.")
    else:
        missing_files = [
            file for file in converted_files if file not in output_files
        ]
        print("Some files are missing after conversion:", missing_files)

def normalize_audio(audio_path):
    #Normalize audio
    audio, sr = librosa.load(audio_path, sr=44100)
    audio_normalized = librosa.util.normalize(audio)
    return audio_normalized, sr

def load_and_normalize_to_matrix(folder_path):
    """
    input: folder_path: path to the folder containing .wav files

    Loads and normalizes all .wav files in a folder and stores them in a matrix.
    """
    audio_data = []
    max_length = 0

    # Normalize all .wav files and find the maximum length
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.wav'):
            input_path = os.path.join(folder_path, file_name)
            try:
                normalized_audio, sr = normalize_audio(input_path)
                audio_data.append(normalized_audio)
                max_length = max(max_length, len(normalized_audio))
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Pad all audio data to the maximum length
    pad_data = [np.pad(data, (0, max_length - len(data))) for data in audio_data]
                        
    audio_matrix = np.concatenate(pad_data)

    return audio_matrix, sr

def segment_audio(audio, sr, segment_length=3, silence_threshold=0.01):
    """
    Segments audio into fixed-length chunks, excluding silent segments.
    """
    segment_length = int(segment_length * sr)
    segments = []
    for start in range(0, len(audio), segment_length):
        segment = audio[start:start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))

        # Calculate RMS energy
        rms = np.sqrt(np.mean(segment**2))
        if rms > silence_threshold:
            segments.append(segment)

    return segments

def feature_extraction(audio_normalized, sr):
    """
    input: audio_normalized: normalized audio data
    Extract four different features from audio data

    return:
    a single feature vector.
    """

    # Mel-frequency cepstral coefficients
    mfcc = np.mean(librosa.feature.mfcc(y=audio_normalized, sr=sr, n_mfcc=13), axis=1)
    # chromagram from a waveform
    chroma = librosa.feature.chroma_stft(y=audio_normalized, sr=sr)
    # Spectral contrast, energy contrast between peaks and valleys
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_normalized, sr=sr), axis=1)
    # Mel is a linear transformation matrix to project FFT bins onto mel-frequency bins
    mel = librosa.feature.melspectrogram(y=audio_normalized, sr=sr)

    #calculate the mel mean
    mel_mean = np.mean(mel, axis=1)
    chroma_mean = np.mean(chroma, axis=1)


    #ad all features together
    features = np.concatenate([mfcc, chroma_mean, spectral_contrast, mel_mean])

    return features, mel, chroma

def classifier(features, classes):
    """
    Inputs:
    - features: Feature matrix (X)
    - classes: Target labels (y)

    Output:
    - Accuracy, precision, and recall scores
    """

    # Split the training data
    x_train, x_test, y_train, y_test = train_test_split(features, classes, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize the SVM classifier
    svm = SVC(kernel='linear', C=1, random_state=42)

    # Train the model
    svm.fit(x_train, y_train)

    # Predict
    y_prediction = svm.predict(x_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction)
    recall = recall_score(y_test, y_prediction)

    # Print results
    print("")
    print("Model Metrics:")
    print("-"*25)
    print(f"Accuracy: {accuracy:.2f} (100% correct predictions)")
    print(f"Precision: {precision:.2f} (False positives rate: {1-precision:.2f})")
    print(f"Recall: {recall:.2f} (False negatives rate: {1-recall:.2f})\n")

    #Print the results againsta a random baseline
    random_predictions = np.random.randint(0, 2, size=len(classes))
    random_accuracy = accuracy_score(classes, random_predictions)
    random_precision = precision_score(classes, random_predictions)
    random_recall = recall_score(classes, random_predictions)


    print("Random Baseline Metrics:")
    print("-"*25)
    print(f"Accuracy: {random_accuracy:.2f} ({random_accuracy*100:.2f}% correct predictions)")
    print(f"Precision: {random_precision:.2f} (False positive rate: {1-random_precision:.2f})")
    print(f"Recall: {random_recall:.2f}" + f" (False negative rate: {1-random_recall:.2f})\n")

    scores = cross_val_score(svm, features, classes, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.2f}\n")

    return {"model": svm, "scaler": scaler}

def validate_model(features_validation, validation_classes, model, scaler):
    """
    Evaluate the trained model on validation data.
    """
    # Scale the validation features
    features_validation_scaled = scaler.transform(features_validation)

    # Predict and evaluate
    validation_predictions = model.predict(features_validation_scaled)
    accuracy = accuracy_score(validation_classes, validation_predictions)
    precision = precision_score(validation_classes, validation_predictions)
    recall = recall_score(validation_classes, validation_predictions)


    print("Validation Metrics:")
    print("-"*25)
    print(f"Accuracy: {accuracy:.2f} ({accuracy*100:.2f}% correct predictions)")
    print(f"Precision: {precision:.2f} (False positive rate: {1-precision:.2f})")
    print(f"Recall: {recall:.2f}" + f" (False negative rate: {1-recall:.2f})\n")

def plot_spectrogram(audio_normalized, sr, title="Spectrogram"):
    """
    Plot the spectrogram of the given audio data.
    """
   
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_combined_histogram(data_car, data_tram, bins, title, xlabel, ylabel, xlim=None):
    """
    Helper function to plot combined histogram for car and tram audio data.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot car data in green
    plt.hist(data_car, bins=bins, color='green', alpha=0.6, label="Car", edgecolor='black')

    # Plot tram data in blue
    plt.hist(data_tram, bins=bins, color='blue', alpha=0.6, label="Tram", edgecolor='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()
    plt.show()

def combine_mfccs(segments, sr):
    """
    Combines MFCC coefficients across all segments for a given class.
    """
    combined_mfccs = []
    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        combined_mfccs.append(mfcc)
    return np.hstack(combined_mfccs)  # Combine along the time axis

def plot_combined_mfcc(mfcc_car, mfcc_tram, sr, title="Combined MFCC Coefficients"):
    """
    Plots the combined MFCC coefficients for car and tram classes.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot the MFCC for car class
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc_car, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Car: MFCC Coefficients")
    plt.ylabel("MFCC Coefficients")

    # Plot the MFCC for tram class
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfcc_tram, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Tram: MFCC Coefficients")
    plt.xlabel("Time (frames)")
    plt.ylabel("MFCC Coefficients")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    """
    Compute and plot the average spectral contrast for car and tram audio files.

    Inputs:
    - dir_audio_car: Directory containing car audio files.
    - dir_audio_tram: Directory containing tram audio files.
    - sr: Sampling rate for audio processing.

    Output:
    - Combined spectral contrast plot for car and tram audio.
    """
    
def plot_combined_mfcc_class(mfcc_car, mfcc_tram, sr, n_mfcc=13):
    """
    Plot combined MFCCs for car and tram classes.

    Args:
    - mfcc_car: MFCC matrix for car class.
    - mfcc_tram: MFCC matrix for tram class.
    - sr: Sampling rate.
    - n_mfcc: Number of MFCC coefficients.
    """
    plt.figure(figsize=(12, 6))

    # Mean across time (columns of MFCC matrix)
    mfcc_car_mean = np.mean(mfcc_car, axis=1)
    mfcc_tram_mean = np.mean(mfcc_tram, axis=1)

    # Standard deviation across time
    mfcc_car_std = np.std(mfcc_car, axis=1)
    mfcc_tram_std = np.std(mfcc_tram, axis=1)

    # Plot for cars
    plt.errorbar(range(1, n_mfcc + 1), mfcc_car_mean, yerr=mfcc_car_std, label="Car", fmt='-o', color="green")

    # Plot for trams
    plt.errorbar(range(1, n_mfcc + 1), mfcc_tram_mean, yerr=mfcc_tram_std, label="Tram", fmt='-o', color="blue")

    # Labels and formatting
    plt.title("Combined MFCCs for Car and Tram Classes")
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("MFCC Values (Mean ± Std Dev)")
    plt.xticks(range(1, n_mfcc + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function
    """
    
    #For folder conversion
    #folder_convert(dir_input_car, dir_audio_output_car)
    #folder_convert(dir_input_tram, dir_audio_output_tram)

    #For validation data conversion
    #folder_convert(dir_validation_input_tram, dir_validation_output_tram)


    # Load and normalize all audio data
    normalized_car, sr = load_and_normalize_to_matrix(dir_audio_output_car)
    normalized_tram, sr = load_and_normalize_to_matrix(dir_audio_output_tram)
    normalized_car_validation, sr = load_and_normalize_to_matrix(dir_validation_output_car)
    normalized_tram_validation, sr = load_and_normalize_to_matrix(dir_validation_output_tram)

    # Segment the audio data
    segment_car = segment_audio(normalized_car, sr, segment_length=3)
    segment_tram = segment_audio(normalized_tram, sr, segment_length=3)
    segment_car_validation = segment_audio(normalized_car_validation, sr, segment_length=3)
    segment_tram_validation = segment_audio(normalized_tram_validation, sr, segment_length=3)

    # Combine MFCCs for each class
    mfcc_car = combine_mfccs(segment_car, sr)
    mfcc_tram = combine_mfccs(segment_tram, sr)

    # Plot the combined MFCCs
    plot_combined_mfcc(mfcc_car, mfcc_tram, sr)
    plot_combined_mfcc_class(mfcc_car, mfcc_tram, sr, n_mfcc=13)

    # Plot the mel-spectrogram differences
    plot_mel_difference(mel_car, mel_tram)

    #Exctract features
    features_car = np.array([feature_extraction(segment, sr) for segment in segment_car])    
    features_tram = np.array([feature_extraction(segment, sr) for segment in segment_tram])
    features_car_validation = np.array([feature_extraction(segment, sr) for segment in segment_car_validation])
    features_tram_validation = np.array([feature_extraction(segment, sr) for segment in segment_tram_validation])

    # Combine the features and classes
    features = np.vstack([features_car, features_tram])
    features_validation = np.vstack([features_car_validation, features_tram_validation])

    # 0 for car, 1 for tram
    classes = np.concatenate([np.zeros(len(features_car)), np.ones(len(features_tram))])
    classes_validation = np.concatenate([np.zeros(len(features_car_validation)), np.ones(len(features_tram_validation))])

    # Train the classifier
    results = classifier(features, classes)

    # Validate the model
    validate_model(features_validation, classes_validation, results["model"], results["scaler"])

if __name__ == "__main__":
    main()
