# Results from the 300-epoch run

Ultralytics saved these automatically during training:

- `results.png`: loss and accuracy curves for every epoch
- `results.csv`: the same numbers as a table
- `confusion_matrix_normalized.png`: which classes the model confuses with each other
- `PR_curve.png`: precision-recall curve for each class on the validation images
- `labels.jpg`: how many labeled examples each class has
- `val_batch0_labels.jpg` and `val_batch0_pred.jpg`: true labels vs. the model's predictions on validation images
- `args.yaml`: the exact training settings
