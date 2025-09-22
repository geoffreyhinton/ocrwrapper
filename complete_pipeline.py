#!/usr/bin/env python3
"""
Complete Invoice Pipeline: OCR + LayoutML Classification
Combines PaddleOCR (text extraction) + Your trained model (classification)
"""

import os
import json
import time
import numpy as np
import paddle
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OCR instance (cached for speed)
_paddleocr_instance = None

def get_paddleocr_instance():
    """Get cached PaddleOCR instance"""
    global _paddleocr_instance
    
    if _paddleocr_instance is None:
        logger.info("🔄 Initializing PaddleOCR (one-time setup)...")
        try:
            import paddleocr
            _paddleocr_instance = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("✅ PaddleOCR ready!")
        except Exception as e:
            logger.error(f"❌ PaddleOCR failed: {e}")
            return None
    
    return _paddleocr_instance

def load_classification_model():
    """Load your trained LayoutML model for classification"""
    model_path = "./output/epoch_10.pdparams"
    
    if not os.path.exists(model_path):
        logger.warning("❌ Classification model not found")
        return None, []
    
    try:
        # Load your trained model
        checkpoint = paddle.load(model_path)
        model_state_dict = checkpoint['model_state_dict']
        
        # Load class list
        with open("./train_data/wildreceipt_paddleocr/class_list.txt", 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Create model architecture
        import paddle.nn as nn
        
        class InvoiceClassifier(nn.Layer):
            def __init__(self, num_classes=26):
                super(InvoiceClassifier, self).__init__()
                self.conv1 = nn.Conv2D(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2D(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2D(128, 256, 3, padding=1)
                self.classifier = nn.Linear(256, num_classes)
                
            def forward(self, x):
                x = paddle.nn.functional.relu(self.conv1(x))
                x = paddle.nn.functional.max_pool2d(x, 2)
                x = paddle.nn.functional.relu(self.conv2(x))
                x = paddle.nn.functional.max_pool2d(x, 2)
                x = paddle.nn.functional.relu(self.conv3(x))
                x = paddle.nn.functional.max_pool2d(x, 2)
                x = paddle.nn.functional.adaptive_avg_pool2d(x, 1)
                x = paddle.flatten(x, 1)
                return self.classifier(x)
        
        # Load model
        model = InvoiceClassifier()
        model.set_state_dict(model_state_dict)
        model.eval()
        
        logger.info(f"✅ Classification model loaded with {len(classes)} classes")
        return model, classes
        
    except Exception as e:
        logger.error(f"❌ Failed to load classification model: {e}")
        return None, []

def extract_text_with_ocr(image_path):
    """Extract text using PaddleOCR"""
    ocr = get_paddleocr_instance()
    if ocr is None:
        return []
    
    try:
        result = ocr.ocr(image_path)
        
        # Debug: Print the actual OCR result structure
        logger.info(f"🔍 OCR result type: {type(result)}")
        logger.info(f"🔍 OCR result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        # Handle different PaddleOCR output formats
        if not result or len(result) == 0:
            logger.warning("No OCR results returned")
            return []
        
        # Get the first page results
        page_result = result[0]
        
        if not page_result:
            logger.warning("Empty page result from OCR")
            return []
        
        extracted_text = []
        
        # Handle new PaddleOCR result format (OCRResult object)
        if hasattr(page_result, 'get') and 'rec_texts' in page_result:
            # New format: OCRResult object with rec_texts, rec_scores, rec_polys
            texts = page_result['rec_texts']
            scores = page_result['rec_scores']
            polys = page_result['rec_polys']
            
            for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                extracted_text.append({
                    'text': text,
                    'bbox': poly.tolist() if hasattr(poly, 'tolist') else poly,  # Convert numpy array to list
                    'confidence': score
                })
        else:
            # Old format: List of [bbox, [text, confidence]]
            for i, line in enumerate(page_result):
                try:
                    # Handle different line formats
                    if len(line) >= 2:
                        bbox = line[0]  # Bounding box coordinates
                        text_info = line[1]  # Text and confidence
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]  # Extracted text
                            confidence = text_info[1]  # Confidence score
                        else:
                            text = str(text_info)
                            confidence = 1.0
                        
                        extracted_text.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': confidence
                        })
                    else:
                        logger.warning(f"Unexpected line format at index {i}: {line}")
                        
                except Exception as e:
                    logger.warning(f"Error processing OCR line {i}: {e}")
                    continue
        
        logger.info(f"✅ Successfully extracted {len(extracted_text)} text elements")
        return extracted_text
        
    except Exception as e:
        logger.error(f"❌ OCR failed: {e}")
        return []

def classify_text_region(text_bbox, image, classifier_model, classes):
    """Classify a text region using text-based classification"""
    # Import the text-based classifier
    from text_based_classifier import TextFieldClassifier
    
    try:
        text = text_bbox.get('text', '')
        if not text:
            return 'unknown', 0.0
        
        # Use text-based classification instead of image-based
        text_classifier = TextFieldClassifier()
        predicted_class, confidence = text_classifier.classify_text(text)
        
        return predicted_class, confidence
            
    except Exception as e:
        logger.error(f"❌ Classification failed: {e}")
        return 'unknown', 0.0

def complete_invoice_pipeline(image_path):
    """Complete pipeline: OCR + Classification"""
    logger.info(f"🚀 Processing: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image not found: {image_path}")
        return None
    
    start_time = time.time()
    
    # Step 1: Load models
    classifier_model, classes = load_classification_model()
    
    # Step 2: Extract text with OCR
    logger.info("📝 Step 1: Extracting text with OCR...")
    extracted_text = extract_text_with_ocr(image_path)
    
    if not extracted_text:
        logger.error("❌ No text extracted")
        return None
    
    logger.info(f"✅ Extracted {len(extracted_text)} text elements")
    
    # Step 3: Classify each text element
    logger.info("🤖 Step 2: Classifying text regions...")
    
    image = Image.open(image_path).convert('RGB')
    
    classified_results = []
    for item in extracted_text:
        # Classify this text region
        predicted_class, class_confidence = classify_text_region(
            item, image, classifier_model, classes
        )
        
        classified_results.append({
            'text': item['text'],
            'bbox': item['bbox'],
            'ocr_confidence': item['confidence'],
            'predicted_class': predicted_class,
            'class_confidence': class_confidence
        })
    
    # Step 4: Organize results
    organized_results = {}
    for item in classified_results:
        class_name = item['predicted_class']
        if class_name not in organized_results:
            organized_results[class_name] = []
        organized_results[class_name].append(item)
    
    processing_time = time.time() - start_time
    
    return {
        'image': os.path.basename(image_path),
        'processing_time': processing_time,
        'total_text_elements': len(extracted_text),
        'classified_elements': classified_results,
        'organized_by_class': organized_results,
        'summary': {
            'classes_found': list(organized_results.keys()),
            'total_classes': len(organized_results)
        }
    }

def main():
    """Main execution"""
    print("🚀 COMPLETE INVOICE PIPELINE")
    print("OCR Text Extraction + LayoutML Classification")
    print("=" * 50)
    
    # Find test images
    test_dir = "./test_images"
    images = []
    
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(test_dir, f))
    
    if not images:
        print("❌ No test images found in ./test_images/")
        return
    
    for i, image_path in enumerate(images):
        print(f"\n📄 Processing image {i+1}/{len(images)}")
        
        result = complete_invoice_pipeline(image_path)
        
        if result:
            print(f"✅ Completed in {result['processing_time']:.2f} seconds")
            print(f"📝 Extracted {result['total_text_elements']} text elements")
            print(f"🏷️  Found {result['summary']['total_classes']} different field types")
            
            print("\n📋 Sample Results:")
            for class_name, items in result['organized_by_class'].items():
                if items and class_name != 'Ignore':
                    item = items[0]  # Show first item of each class
                    print(f"   {class_name}: '{item['text']}' (conf: {item['class_confidence']:.2f})")
            
            # Save results
            output_file = f"./output/complete_pipeline_{os.path.basename(image_path)}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Detailed results saved to: {output_file}")
        else:
            print("❌ Failed to process")

if __name__ == "__main__":
    main()