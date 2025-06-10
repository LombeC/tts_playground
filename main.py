import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import time
def chatterbox_tts():
    print("Welcome to the Chatterbox TTS system!")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    text = "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
    # wav = model.generate(text)
    # ta.save("test-1.wav", wav, model.sr)
    # If you want to synthesize with a different voice, specify the audio prompt
    exaggeration=0.5
    cfg_weight=0.3
    audio_prompt_path="target.wav"  # Path to your audio prompt file
    model.prepare_conditionals(audio_prompt_path)
    start_time = time.time()
    wav = model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight)
    ta.save("test2.wav", wav, model.sr)
    print(f"Audio generation took {time.time() - start_time:.2f} seconds")

def main():
    print("Hello from tts-playground!")
    chatterbox_tts()

if __name__ == "__main__":
    main()
