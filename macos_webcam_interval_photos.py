import subprocess
import time
from datetime import datetime
import os
import base64
import openai
import requests
from dotenv import load_dotenv
import pygame
import cv2

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ElevenLabs API key
elevenlabs_api_key = os.getenv("ELEVENLABS_KEY")

# Initialize pygame mixer
pygame.mixer.init()

# Directory to store audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def list_cameras():
    """List available cameras using imagesnap."""
    result = subprocess.run(['imagesnap', '-l'], capture_output=True, text=True)
    cameras = result.stdout.strip().split('\n')[1:]  # Skip the first line which is a header
    
    if not cameras:
        print("No cameras found.")
        return None
    
    if len(cameras) == 1:
        print(f"Only one camera found: {cameras[0]}. Using this camera.")
        return cameras[0].strip()
    
    print("Available cameras:")
    for i, camera in enumerate(cameras):
        print(f"{i + 1}. {camera}")
    
    while True:
        choice = input("Enter the number of the camera you want to use: ")
        try:
            choice = int(choice) - 1
            if 0 <= choice < len(cameras):
                return cameras[choice].strip()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def take_photo(camera):
    """Take a photo using the specified camera with imagesnap."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    
    # Remove the "=> " prefix if it exists
    camera_name = camera[3:] if camera.startswith("=> ") else camera
    
    result = subprocess.run(['imagesnap', '-d', camera_name, filename], capture_output=True, text=True)
    
    if os.path.exists(filename):
        print(f"Photo saved as {filename}")
        if has_people(filename):
            print("People detected in the image. Analyzing...")
            analyze_image(filename)
            return True
        else:
            print("No people detected in the image. Skipping analysis.")
            os.remove(filename)  # Optionally remove the photo without people
            return False
    else:
        print(f"Error: Could not capture image. {result.stderr}")
        return False

def generate_audio(text, voice_id):
    """Generate audio from text using ElevenLabs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "language_code": "en",  # Changed to English for this example
        "voice_settings": {
            "stability": 0.31,
            "similarity_boost": 0.97,
            "style": 0.50,
            "use_speaker_boost": True
        },
        "seed": 123,
    }
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_api_key
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        audio_file = os.path.join(AUDIO_DIR, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3")
        with open(audio_file, 'wb') as f:
            f.write(response.content)
        print(f"Created new audio file: {audio_file}")
        return audio_file
    else:
        print(f"Error generating audio: {response.status_code}, {response.text}")
        return None

def play_audio(audio_file):
    """Play the generated audio file."""
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def process_dialog(dialog):
    """Process the dialog, generate voices, and play them."""
    lines = dialog.strip().split('\n')
    
    knotan_voice = "f2yUVfK5jdm78zlpcZ8C"
    skallan_voice = "9yzdeviXkFddZ4Oz8Mok"
    
    for line in lines:
        if line.startswith("Knotan:"):
            voice_id = knotan_voice
            text = line.replace("Knotan:", "").strip()
        elif line.startswith("Skallan:"):
            voice_id = skallan_voice
            text = line.replace("Skallan:", "").strip()
        else:
            continue  # Skip lines that don't start with either prefix
        
        print(f"Processing line: {text}")
        
        audio_file = generate_audio(text, voice_id)
        if audio_file:
            play_audio(audio_file)
        
        time.sleep(0.5)

def analyze_image(image_path):
    """Analyze the image using OpenAI's vision model and process the resulting dialog."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image detailed. Pay extra attention to the people \
					in the image and what they are doing."},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    description = response.choices[0].message.content


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Imagine that you are two skeletons (Knotan och Skallan) in a garden\
						during halloween and you are watching the scene described below. \
						let the skeletons comment what they see and call on the people they are abnalyuzing like: you therein the green hoodie.\
						as in the style of Statler and Waldorf in the muppets but keep it short and catchy and funny and in swedish max four sentences? never use markup. \
                        Each line xtarst with the ane of the skelleton talking  like Skallan: \
                      Here is the description of the scene: " + description},
						
                ],
            }
        ],
        max_tokens=300,
    )


    print("Image Analysis and Dialog Generation:")
    dialog = response.choices[0].message.content
    print(dialog)

    # Process the generated dialog
    process_dialog(dialog)

def has_people(image_path):
    """Check if the image contains people using face detection."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def main():
    camera = list_cameras()
    if camera is None:
        return
    
    print(f"Selected camera: {camera}")
    interval = int(input("Enter the interval between photos (in seconds): "))
    
    try:
        while True:
            if take_photo(camera):
                time.sleep(interval)
            else:
                print("Error taking photo. Retrying in 5 seconds...")
                time.sleep(5)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")

if __name__ == "__main__":
    main()
