# Data Folder

Place your CSV training data files here.

## Expected Format

Your CSV should have these columns:
- `feature_1` through `feature_10` (10 features)
- `target` (binary classification target: 0 or 1)

## Example

```csv
feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,target
0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0
0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,1
```

## Usage

```bash
python -m flclient train --data data/your_file.csv --join-code ABC123
``` 