from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib

# Step 1: Load Model, Tokenizer, and Label Encoder
model = load_model('emotion_model.h5')
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define max_length based on training
max_length = model.input_shape[1]

# Step 2: Define Inference Function
def predict_emotion(text, tokenizer, model, max_length, label_encoder):
    """
    Predict the emotion of a given text input.

    Args:
        text (str): Input text to classify.
        tokenizer (Tokenizer): Fitted tokenizer for preprocessing.
        model (Sequential): Trained Keras model for prediction.
        max_length (int): Maximum length used during model training.
        label_encoder (LabelEncoder): Encoder for decoding emotion labels.

    Returns:
        str: Predicted emotion label.
    """
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    # Predict emotion
    prediction = model.predict(padded_sequence)
    emotion_index = prediction.argmax()  # Get the index of the highest probability
    emotion_label = label_encoder.inverse_transform([emotion_index])[0]  # Decode to emotion label
    return emotion_label

# Step 3: Define Function to Show YouTube Link
def show_song(emotion):
    """
    Display a YouTube link based on the emotion.

    Args:
        emotion (str): Predicted emotion label.

    Returns:
        None
    """
    emotion_songs = {
        "fear": "https://www.youtube.com/watch?v=fear_song_link",
        "joy": "https://www.youtube.com/watch?v=joy_song_link",
        "sadness": "https://www.youtube.com/watch?v=sadness_song_link",
        "anger": "https://www.youtube.com/watch?v=anger_song_link"
    }
    link = emotion_songs.get(emotion, "No song available for this emotion.")
    print(f"Recommended song for your current situation: {link}")

# Step 4: Get Input from User
if __name__ == "__main__":
    while True:
        user_text = input("Enter a text to analyze emotion (or type 'exit' to quit): ")
        if user_text.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break
        predicted_emotion = predict_emotion(user_text, tokenizer, model, max_length, label_encoder)
        print(f"Predicted Emotion: {predicted_emotion}")
        show_song(predicted_emotion)
