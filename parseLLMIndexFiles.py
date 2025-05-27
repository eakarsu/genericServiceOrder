#!/usr/bin/env python3
"""
Parse attached files and extract all sector prompt files
Creates individual .txt files for each sector's prompts and rules
"""

import os
import re
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = ["sector_prompts", "sector_rules"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def parse_file(filename, sectors_map):
    """Parse a specific file and extract sector content"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n🔍 Parsing {filename}...")
        
        # Extract all sectors from this file
        for sector_name, file_patterns in sectors_map.items():
            for file_type, pattern in file_patterns.items():
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    # Clean up the extracted content
                    extracted_content = matches[0].strip()
                    
                    # Remove the markdown code block markers if present
                    if extracted_content.startswith("```"):
                        lines = extracted_content.split('\n')
                        if lines.startswith('```'):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        extracted_content = '\n'.join(lines)
                    
                    # Determine output filename and directory
                    if file_type in ['prompt', 'prompt2', 'prompt3']:
                        if file_type == 'prompt':
                            output_file = f"sectors/{sector_name}/prompt.txt"
                        else:
                            output_file = f"sectors/{sector_name}/{file_type}.txt"
                    else:  # rules
                        output_file = f"sectors/{sector_name}/rules.txt"
                    
                    # Write the extracted content
                    write_file(output_file, extracted_content)
                    print(f"✅ Extracted {sector_name}_{file_type} from {filename}")
                
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
    except Exception as e:
        print(f"❌ Error parsing {filename}: {e}")

def write_file(filename, content):
    """Write content to file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📝 Created: {filename}")
    except Exception as e:
        print(f"❌ Error writing {filename}: {e}")

def extract_all_sectors():
    """Extract all sector files from the attached documents"""
    
    # Define patterns for each sector and file type
    sectors_patterns = {
        # Healthcare (p1.txt)
        "healthcare": {
            "prompt": r"### \*\*healthcare_prompt\.txt\*\*.*?\n```\n(.*?)\n```",
            "prompt2": r"### \*\*healthcare_prompt2\.txt\*\*.*?\n```\n(.*?)\n```",
            "rules": r"### \*\*healthcare_rules\.txt\*\*.*?\n```\n(.*?)\n```",
            "prompt3": r"### \*\*healthcare_prompt3\.txt\*\*.*?\n```\n(.*?)\n```"
        },
        
        # Real Estate (p1.txt)
        "real_estate": {
            "prompt": r'### \*\*real_estate_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*real_estate_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*real_estate_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*real_estate_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Auto Repair (p2.txt)
        "auto_repair": {
            "prompt": r'### \*\*auto_repair_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*auto_repair_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*auto_repair_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*auto_repair_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Beauty Salon (p2.txt)
        "beauty_salon": {
            "prompt": r'### \*\*beauty_salon_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*beauty_salon_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*beauty_salon_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*beauty_salon_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Legal Services (p2.txt)
        "legal_services": {
            "prompt": r'### \*\*legal_services_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*legal_services_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*legal_services_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*legal_services_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Financial Services (p2.txt)
        "financial_services": {
            "prompt": r'### \*\*financial_services_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*financial_services_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*financial_services_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*financial_services_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Transportation (p3.txt)
        "transportation": {
            "prompt": r'### \*\*transportation_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*transportation_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*transportation_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*transportation_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Travel Hotel (p3.txt)
        "travel_hotel": {
            "prompt": r'### \*\*travel_hotel_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*travel_hotel_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*travel_hotel_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*travel_hotel_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Home Services (p4.txt)
        "home_services": {
            "prompt": r'### \*\*home_services_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*home_services_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*home_services_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*home_services_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Education Tutoring (p4.txt)
        "education_tutoring": {
            "prompt": r'### \*\*education_tutoring_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*education_tutoring_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*education_tutoring_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*education_tutoring_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Insurance (p4.txt)
        "insurance": {
            "prompt": r'### \*\*insurance_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*insurance_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*insurance_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*insurance_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Event Planning (p4.txt)
        "event_planning": {
            "prompt": r'### \*\*event_planning_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*event_planning_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*event_planning_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*event_planning_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Fitness Gym (p5.txt)
        "fitness_gym": {
            "prompt": r'### \*\*fitness_gym_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*fitness_gym_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*fitness_gym_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*fitness_gym_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Photography (p5.txt)
        "photography": {
            "prompt": r'### \*\*photography_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*photography_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*photography_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*photography_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        },
        
        # Pet Services (p5.txt)
        "pet_services": {
            "prompt": r'### \*\*pet_services_prompt\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt2": r'### \*\*pet_services_prompt2\.txt\*\*.*?\n```\n(.*?)\n```',
            "rules": r'### \*\*pet_services_rules\.txt\*\*.*?\n```\n(.*?)\n```',
            "prompt3": r'### \*\*pet_services_prompt3\.txt\*\*.*?\n```\n(.*?)\n```',
        }
    }
    
    # Create output directories
    create_directories()
    
    # Map files to sectors they contain
    file_sector_map = {
        "p1.txt": ["healthcare", "real_estate"],
        "p2.txt": ["auto_repair", "beauty_salon", "legal_services", "financial_services"], 
        "p3.txt": ["transportation", "travel_hotel"],
        "p4.txt": ["home_services", "education_tutoring", "insurance", "event_planning"],
        "p5.txt": ["fitness_gym", "photography", "pet_services"]
    }
    
    total_files_created = 0
    
    # Parse each file
    for filename, sector_list in file_sector_map.items():
        if os.path.exists(filename):
            # Filter patterns for sectors in this file
            file_patterns = {sector: sectors_patterns[sector] for sector in sector_list if sector in sectors_patterns}
            parse_file(filename, file_patterns)
            total_files_created += len(sector_list) * 4  # 4 files per sector
        else:
            print(f"⚠️ File {filename} not found")
    
    print(f"\n🎉 Extraction complete! Expected {total_files_created} files created.")
    
    # Summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"💾 Extracted content from: {len(file_sector_map)} files")
    print(f"🏢 Total sectors: {len(sectors_patterns)}")
    print(f"📁 Files per sector: 4 (prompt.txt, prompt2.txt, rules.txt, prompt3.txt)")
    print(f"📂 Output directories: sector_prompts/, sector_rules/")

def verify_extraction():
    """Verify that all expected files were created"""
    print(f"\n🔍 VERIFICATION:")
    
    expected_sectors = [
        "healthcare", "real_estate", "auto_repair", "beauty_salon", 
        "legal_services", "financial_services", "transportation", "travel_hotel",
        "home_services", "education_tutoring", "insurance", "event_planning",
        "fitness_gym", "photography", "pet_services"
    ]
    
    missing_files = []
    created_files = []
    
    for sector in expected_sectors:
        # Check for 4 files per sector
        files_to_check = [
            f"sectors/{sector}/prompt.txt",
            f"sectors/{sector}/prompt2.txt", 
            f"sectors/{sector}/prompt3.txt",
            f"sectors/{sector}/rules.txt"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                created_files.append(file_path)
            else:
                missing_files.append(file_path)
    
    print(f"✅ Created files: {len(created_files)}")
    print(f"❌ Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
    
    if created_files:
        print(f"\n✅ Successfully created:")
        for file in created_files[:10]:  # Show first 10
            print(f"   - {file}")
        if len(created_files) > 10:
            print(f"   ... and {len(created_files) - 10} more files")

if __name__ == "__main__":
    print("🚀 Universal Sector File Parser")
    print("=" * 50)
    
    extract_all_sectors()
    verify_extraction()
    
    print(f"\n✨ All sector files ready for your Universal Service Bot!")
