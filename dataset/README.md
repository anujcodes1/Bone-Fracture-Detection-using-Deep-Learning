# Dataset

Place your X-ray dataset here using this structure:

- `train/fractured`
- `train/non_fractured`
- `val/fractured`
- `val/non_fractured`
- `test/fractured`
- `test/non_fractured`

Add the X-ray images into those folders before training the model.

For the public FracAtlas dataset, you can automatically organize the extracted files by running:

- `python dataset\organize_fracatlas.py "C:\path\to\FracAtlas"`
- or `organize_fracatlas.bat "C:\path\to\FracAtlas"`

The organizer uses `dataset.csv` to build a binary `fractured` vs `non_fractured` split that matches this project's classifier.
