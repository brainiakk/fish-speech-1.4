import os

import pygame

from modules.fish_speech.tools.llama.generate import main as generate
from pathlib import Path
from modules.fish_speech.tools.vqgan.inference import main as infer

class VoiceService:
    def __init__(self):
        self._output_dir = "outputs/"
        os.makedirs(self._output_dir, exist_ok=True)

    def fishspeech(self, text):
        infer(input_path=Path("jarvis.wav"), output_path=Path(self._output_dir+"fake.wav"),
              checkpoint_path="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
              config_name="firefly_gan_vq", device="cpu")

        generate(text=text,
                 prompt_text=[
                     # "No, he's going to die. I was making the moment more epic. Leprechauns are tiny, green, and Irish, and that is offensive. No, he's going to die. Will explain everything if you'll kindly come with me. Yes, my lord, like making beds. Or cooking food, polishing the silver. I am trying, my lord. Prefer the word sociopath.Clear for now. Hands off. There is a time and a place for everything. This is not the time, nor the place.",
                     "accessing alarm and interface settings in this window you can set up your customized greeting and alarm preferences the world needs your expertise or at least your presence launching a series of displays to help guide you",
                     ],
                 prompt_tokens=[Path(self._output_dir+"fake.npy")],
                 checkpoint_path=Path("checkpoints/fish-speech-1.4"),
                 half=False,
                 device="cpu",
                 num_samples=2,
                 max_new_tokens=1024,
                 top_p=0.7,
                 repetition_penalty=1.2,
                 temperature=0.3,
                 compile=True,
                 seed=42,
                 iterative_prompt=True,
                 chunk_length=100)


        infer(input_path=Path("codes_1.npy"), output_path=Path(self._output_dir+"output.wav"),
              checkpoint_path="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
              config_name="firefly_gan_vq", device="cpu")

        self.play(self._output_dir+"output.wav")

    def play(self, temp_audio_file):
        pygame.mixer.quit()
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.stop()
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

        # os.remove(temp_audio_file)
