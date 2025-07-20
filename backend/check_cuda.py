import torch

print(torch.__version__)          # e.g. '2.1.2+cu118'
print(torch.version.cuda)         # e.g. '11.8'
print(torch.cuda.is_available())  # Should be True

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available")

    