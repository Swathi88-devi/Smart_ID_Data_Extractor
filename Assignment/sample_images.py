"""
Generate 20 Clear, Sharp Sample ID Card Images
This script creates realistic ID cards with READABLE TEXT for testing
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create input_images folder
os.makedirs('input_images', exist_ok=True)

# Sample data
FIRST_NAMES = ['John', 'Sarah', 'Michael', 'Emma', 'David', 'Lisa', 'James', 'Anna', 
               'Robert', 'Maria', 'William', 'Jessica', 'Richard', 'Amanda', 'Joseph']
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
              'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson']
COMPANIES = ['TechCorp', 'DataFlow', 'CloudSys', 'InfoWave', 'ByteWorks', 'NetSync', 'CodeHub', 'DigitalX']
DEPARTMENTS = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations', 'Design', 'Support']

def generate_sample_images(num_images=20):
    """Generate sharp, clear ID card images with readable text"""
    
    for i in range(1, num_images + 1):
        # Create a white image (800x600 pixels)
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use default font, fallback to basic font
        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            label_font = ImageFont.truetype("arial.ttf", 24)
            value_font = ImageFont.truetype("arial.ttf", 22)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            value_font = ImageFont.load_default()
        
        # Generate random data
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"
        
        email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(COMPANIES).lower()}.com"
        company = random.choice(COMPANIES)
        department = random.choice(DEPARTMENTS)
        phone = f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
        emp_id = f"EMP-{random.randint(10000, 99999)}"
        
        # Draw blue header
        draw.rectangle([(0, 0), (800, 120)], fill='#1e3a8a')
        
        # Draw "ID CARD" title in white
        draw.text((300, 40), "ID CARD", fill='white', font=title_font)
        
        # Draw content with labels and values
        y_position = 150
        line_spacing = 70
        
        # NAME
        draw.text((50, y_position), "NAME:", fill='black', font=label_font)
        draw.text((250, y_position), full_name, fill='darkblue', font=value_font)
        
        # EMAIL
        y_position += line_spacing
        draw.text((50, y_position), "EMAIL:", fill='black', font=label_font)
        draw.text((250, y_position), email, fill='darkblue', font=value_font)
        
        # COMPANY
        y_position += line_spacing
        draw.text((50, y_position), "COMPANY:", fill='black', font=label_font)
        draw.text((250, y_position), company, fill='darkblue', font=value_font)
        
        # DEPARTMENT
        y_position += line_spacing
        draw.text((50, y_position), "DEPARTMENT:", fill='black', font=label_font)
        draw.text((250, y_position), department, fill='darkblue', font=value_font)
        
        # PHONE
        y_position += line_spacing
        draw.text((50, y_position), "PHONE:", fill='black', font=label_font)
        draw.text((250, y_position), phone, fill='darkblue', font=value_font)
        
        # EMP ID
        y_position += line_spacing
        draw.text((50, y_position), "EMP ID:", fill='black', font=label_font)
        draw.text((250, y_position), emp_id, fill='darkblue', font=value_font)
        
        # Draw border
        draw.rectangle([(10, 10), (790, 590)], outline='black', width=3)
        
        # Save image
        filename = f'input_images/sample_{i:02d}.jpg'
        img.save(filename, 'JPEG', quality=95)
        print(f"Created: {filename}")
        print(f"  Name: {full_name}")
        print(f"  Email: {email}")
        print()

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING 20 CLEAR SAMPLE ID CARD IMAGES")
    print("=" * 60)
    print()
    
    generate_sample_images(20)
    
    print("=" * 60)
    print("DONE! 20 images created in 'input_images/' folder")
    print("=" * 60)
