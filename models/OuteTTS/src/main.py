import outetts

# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

# Print available default speakers
interface.print_default_speakers()

# Load a default speaker
speaker = interface.load_default_speaker(name="male_1")

# Generate speech
output = interface.generate(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    speaker=speaker,
)

# Save the generated speech to a file
output.save("output.wav")
