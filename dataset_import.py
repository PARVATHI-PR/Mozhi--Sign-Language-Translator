import kagglehub

# Download latest version
path = kagglehub.dataset_download("drblack00/isl-csltr-indian-sign-language-dataset")

print("Path to dataset files:", path)