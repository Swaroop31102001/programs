import speech_recognition as sr
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write


# Function to apply noise reduction using a bandpass filter
def reduce_noise(audio_data, sample_rate):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    filtered_audio = bandpass_filter(audio_data, 300, 3400, sample_rate, order=6)
    return np.int16(filtered_audio)


# Function to recognize speech from the microphone
def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    print("Initializing microphone...")

    with sr.Microphone(sample_rate=16000) as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Listening for speech...")

        try:
            audio = recognizer.listen(source, timeout=15)  # Listen for 15 seconds max
            print("Processing and reducing noise...")

            # Convert audio to numpy for noise reduction
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            sample_rate = source.SAMPLE_RATE
            reduced_audio = reduce_noise(audio_data, sample_rate)

            # Save filtered audio to a WAV file (optional for debugging)
            write("filtered_audio.wav", sample_rate, reduced_audio)

            # Use Google Web Speech API for recognition
            print("Recognizing speech...")
            text = recognizer.recognize_google(audio)
            print(f"Speech Recognized: {text}")

        except sr.UnknownValueError:
            print("Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            print(f"API error occurred: {e}")
        except Exception as ex:
            print(f"An error occurred: {ex}")


if __name__ == "__main__":
    recognize_speech_from_microphone()
