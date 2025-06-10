import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
def chatterbox_tts():
    print("Welcome to the Chatterbox TTS system!")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    text = "I can't recite the whole thing word for word, but I'd love to share some key parts or talk about its history if you want.."
    # wav = model.generate(text)
    # ta.save("test-1.wav", wav, model.sr)
    # If you want to synthesize with a different voice, specify the audio prompt
    AUDIO_PROMPT_PATH="target.wav"  # Path to your audio prompt file
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save("test2.wav", wav, model.sr)

def main():
    print("Hello from tts-playground!")
    chatterbox_tts()

if __name__ == "__main__":
    main()
