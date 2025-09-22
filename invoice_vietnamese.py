import paddle
from paddleocr import logger


# Global OCR instance (cached for speed)
_paddleocr_instance = None

def get_paddleocr_instance():
    """Get cached PaddleOCR instance"""
    global _paddleocr_instance
    
    if _paddleocr_instance is None:
        logger.info("🔄 Initializing PaddleOCR (one-time setup)...")
        try:
            import paddleocr
            _paddleocr_instance = paddleocr.PaddleOCR(use_angle_cls=True, lang='vi')
            logger.info("✅ PaddleOCR ready for Vietnamese!")
        except Exception as e:
            logger.error(f"❌ PaddleOCR failed: {e}")
            return None
    return _paddleocr_instance


def extract_text_with_ocr(image_path):
    """Extract text using PaddleOCR"""
    ocr = get_paddleocr_instance()
    if ocr is None:
        return []
    
    try:
        result = ocr.predict(image_path)
        
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

def main():
    """Main execution"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python invoice_vietnamese.py <image_path>")
        return
    
    image_path = sys.argv[1]
    extracted_text = extract_text_with_ocr(image_path)
    
    # Import and use Vietnamese text corrector
    from vietnamese_text_corrector import VietnameseTextCorrector
    corrector = VietnameseTextCorrector()
    
    print("\n" + "="*80)
    print("VIETNAMESE INVOICE OCR RESULTS")
    print("="*80)
    
    corrected_data = corrector.correct_invoice_data(extracted_text)
    
    for item in corrected_data:
        if 'text_original' in item and item['text'] != item['text_original']:
            print(f"✅ CORRECTED:")
            print(f"   Original:  {item['text_original']}")
            print(f"   Corrected: {item['text']}")
            print(f"   BBox: {item['bbox']}, Confidence: {item['confidence']:.2f}")
            print()
        else:
            print(f"Text: {item['text']}, BBox: {item['bbox']}, Confidence: {item['confidence']:.2f}")
    
    # Show correction statistics
    stats = corrector.get_correction_stats(extracted_text)
    print("\n" + "="*80)
    print("CORRECTION STATISTICS")
    print("="*80)
    print(f"Total text elements: {stats['total_items']}")
    print(f"Corrected elements: {stats['corrected_items']}")
    print(f"Correction rate: {stats['correction_rate']:.1%}")
    
    if stats['corrections']:
        print(f"\nDetailed corrections made:")
        for i, correction in enumerate(stats['corrections'], 1):
            print(f"{i}. '{correction['original']}' → '{correction['corrected']}'")

if __name__ == "__main__":
    main()