from tensorflow.keras.models import load_model

model = load_model("../models/brain_tumor_multiclass.keras")

base_model = model.layers[0]

print("\nBASE MODEL LAYERS:\n")

for layer in base_model.layers[-20:]:
    print(layer.name)