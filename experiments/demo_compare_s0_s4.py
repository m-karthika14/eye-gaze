"""Comparison demo: Raw S0 vs S1 vs S4 across noisy gaze sequences."""

import random
from decoder.char_lm import CharLM
from decoder.nearest_key import KeyboardLayout, nearest_key
from experiments.gaze_sequence import key_id_to_char, simulate_gaze_sequence
from experiments.s1 import S1System
from experiments.s4 import S4System

layout = KeyboardLayout.from_json()
lm = CharLM.load_default()

test_phrases = [
    "hello world",
    "i need water",
    "please call the doctor",
    "how are you today",
]

print("=" * 70)
print("  ACCURACY DEMO: S0 (Raw Nearest Key) vs S1 vs S4 (Joint Probabilistic LM)")
print("=" * 70)

for phrase in test_phrases:
    print(f"\nTarget Phrase: '{phrase}'")
    for noise in [0.03, 0.08, 0.14]:
        obs = simulate_gaze_sequence(phrase, layout, noise_sigma=noise, rng=random.Random(42))
        
        # S0: What raw nearest key selects without any LM
        raw_keys = "".join([
            key_id_to_char(nearest_key(o.mu_x, o.mu_y, layout)) if nearest_key(o.mu_x, o.mu_y, layout) else "?"
            for o in obs
        ])
        
        # S1: Hard key + post-hoc LM
        s1 = S1System(layout, lm)
        s1_out = s1.run_sequence(obs)
        
        # S4: Joint Probabilistic LM
        s4 = S4System(layout, lm)
        s4_out = s4.run_sequence(obs)
        
        print(f"  [Gaze Noise sigma = {noise:.2f}]")
        print(f"    S0 (Raw Dwell / In App):    '{raw_keys}'")
        print(f"    S1 (Hard Key + LM Fix):     '{s1_out}'")
        print(f"    S4 (Probabilistic Joint LM): '{s4_out}'")
    print("-" * 70)
