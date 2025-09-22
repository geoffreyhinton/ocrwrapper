#!/usr/bin/env python3
"""
Vietnamese Text Post-Processing for OCR Results
Corrects common Vietnamese OCR errors and improves text recognition
"""

import re
from typing import Dict, List

class VietnameseTextCorrector:
    """Post-process Vietnamese OCR text to fix common errors"""
    
    def __init__(self):
        # Common Vietnamese OCR error corrections
        self.corrections = {
            # Common diacritic errors
            'Tng': 'tặng',
            'Tièn': 'tiền', 
            'Tien': 'tiền',
            'thieu': 'thiếu',
            'chiec': 'chiếc',
            'Chiéc': 'Chiếc',
            'duong': 'đường',
            'Duong': 'Đường',
            'truoc': 'trước',
            'truóc': 'trước',
            'doi': 'đối',
            'dói': 'đối',
            'kiem': 'kiểm',
            'kiêm': 'kiểm',
            'lap': 'lập',
            'lâp': 'lập',
            'nhan': 'nhận',
            'nhân': 'nhận',
            'cuu': 'cứu',
            'cúu': 'cứu',
            'tai': 'tại',
            'tāi': 'tại',
            
            # Company/invoice terms
            'CONG TY': 'CÔNG TY',
            'Cong ty': 'Công ty',
            'CP': 'CP',
            'Ma so': 'Mã số',
            'Mā so': 'Mã số',
            'Dia chi': 'Địa chỉ',
            'Diā chi': 'Địa chỉ',
            'Don vi': 'Đơn vị',
            'Đon vi': 'Đơn vị',
            'Ban hang': 'Bán hàng',
            'Bán hang': 'Bán hàng',
            'Mua hang': 'Mua hàng',
            'Mua hàng': 'Mua hàng',
            'Hang hoa': 'Hàng hóa',
            'Hàng hoa': 'Hàng hóa',
            'Dich vu': 'Dịch vụ',
            'Dịch vu': 'Dịch vụ',
            'So luong': 'Số lượng',
            'Sô luong': 'Số lượng',
            'Don gia': 'Đơn giá',
            'Đon gia': 'Đơn giá',
            'Thanh toan': 'Thanh toán',
            'Thành toan': 'Thành toán',
            'Gia tri': 'Giá trị',
            'Giá tri': 'Giá trị',
            'Gia tang': 'Gia tăng',
            'Gia táng': 'Gia tăng',
            
            # Numbers and currency
            'dong': 'đồng',
            'Dong': 'Đồng',
            'triu': 'triệu',
            'Triu': 'Triệu',
            'trieu': 'triệu',
            'Trieu': 'Triệu',
            'ty': 'tỷ',
            'Ty': 'Tỷ',
            
            # Common Vietnamese words
            'khong': 'không',
            'Khong': 'Không',
            'thang': 'tháng',
            'Thang': 'Tháng',
            'nam': 'năm',
            'Nam': 'Năm',
            'ngay': 'ngày',
            'Ngay': 'Ngày',
            'ngo': 'ngõ',
            'Ngo': 'Ngõ',
            'phuong': 'phường',
            'Phuong': 'Phường',
            'quan': 'quận',
            'Quan': 'Quận',
            'thanh pho': 'thành phố',
            'Thanh pho': 'Thành phố',
            'Ha Noi': 'Hà Nội',
            'Ho Chi Minh': 'Hồ Chí Minh',
            
            # Signature and validation terms
            'hop le': 'hợp lệ',
            'Hop le': 'Hợp lệ',
            'chu ky': 'chữ ký',
            'Chu ky': 'Chữ ký',
            'xac nhan': 'xác nhận',
            'Xac nhan': 'Xác nhận',
        }
        
        # Vietnamese words for context-based corrections
        self.vietnamese_context = {
            'invoice_terms': ['hóa đơn', 'giá trị gia tăng', 'thuế', 'thành tiền', 'tổng tiền'],
            'company_terms': ['công ty', 'doanh nghiệp', 'mã số thuế', 'địa chỉ'],
            'product_terms': ['hàng hóa', 'dịch vụ', 'số lượng', 'đơn giá', 'chiếc'],
            'free_items': ['hàng tặng', 'không thu tiền', 'miễn phí', 'tặng kèm']
        }
    
    def correct_text(self, text: str) -> str:
        """Apply corrections to Vietnamese text"""
        if not text:
            return text
            
        corrected = text
        
        # Apply direct corrections
        for wrong, correct in self.corrections.items():
            corrected = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, corrected, flags=re.IGNORECASE)
        
        # Special handling for "Hàng Tặng Không Thu Tiền"
        corrected = re.sub(r'\b(Hàng\s+)?Tng\s+Không\s+Thu\s+Tièn\b', 'Hàng tặng không thu tiền', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b(Hàng\s+)?tặng\s+Không\s+Thu\s+Tièn\b', 'Hàng tặng không thu tiền', corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def correct_invoice_data(self, extracted_text: List[Dict]) -> List[Dict]:
        """Correct all text elements in extracted invoice data"""
        corrected_data = []
        
        for item in extracted_text:
            corrected_item = item.copy()
            corrected_item['text'] = self.correct_text(item['text'])
            corrected_item['text_original'] = item['text']  # Keep original for comparison
            corrected_data.append(corrected_item)
        
        return corrected_data
    
    def get_correction_stats(self, extracted_text: List[Dict]) -> Dict:
        """Get statistics about corrections made"""
        total_items = len(extracted_text)
        corrected_items = 0
        corrections_made = []
        
        for item in extracted_text:
            original = item['text']
            corrected = self.correct_text(original)
            
            if original != corrected:
                corrected_items += 1
                corrections_made.append({
                    'original': original,
                    'corrected': corrected,
                    'bbox': item['bbox']
                })
        
        return {
            'total_items': total_items,
            'corrected_items': corrected_items,
            'correction_rate': corrected_items / total_items if total_items > 0 else 0,
            'corrections': corrections_made
        }

def main():
    """Test the Vietnamese text corrector"""
    test_texts = [
        "(Hàng Tng Không Thu Tièn)",
        "Dia chi: Só 16, Ngō 247",
        "CONG TY KÉ TOÁN THIÊN UNG",
        "Só tièn chiét kháu:",
        "Tivi Sam Sung 32inchs",
        "Chiéc",
        "Don gia",
        "Thanh toan"
    ]
    
    corrector = VietnameseTextCorrector()
    
    print("Vietnamese OCR Text Correction Test:")
    print("=" * 50)
    
    for text in test_texts:
        corrected = corrector.correct_text(text)
        if text != corrected:
            print(f"Original:  {text}")
            print(f"Corrected: {corrected}")
            print("-" * 30)
        else:
            print(f"No change: {text}")

if __name__ == "__main__":
    main()