# Prepate dataset in train_data 
  dataset:
    name: InvoiceDataset  
    data_dir: train_data/wildreceipt_paddleocr/val
    annotation_file: train_data/wildreceipt_paddleocr/val/val.json
    class_list: train_data/wildreceipt_paddleocr/class_list.txt# Training to create model
python train_simple_layoutml.py -c config_wildreceipt.yml
> Model saved to ./output/epoch_10.pdparams
> [2025-09-22 15:46:04,031] [    INFO] train_simple_layoutml.py:385 - Training completed!

# Evaluate Model
python complete_pipeline.py
