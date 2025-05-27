#!/usr/bin/env python3
"""
Universal Service Bot - Complete multi-sector intent detection and service routing system
Supports 18+ business sectors with intelligent intent recognition
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from datetime import datetime

# Enhanced MenuParser from your existing code
class MenuParser:
    def __init__(self):
        self.categories = []
        self.rules = []
        self.current_category = None
        self.current_rule = None
        self.patterns = {
            'category': re.compile(r'^\[Begin Category\]\s*(.*)'),
            'category_end': re.compile(r'^\[End Category\]'),
            'rule_begin': re.compile(r'^\[Begin Rule\]\s*(.*)'),
            'rule_end': re.compile(r'^\[End Rule\]'),
            'base_price': re.compile(r'Base Price:\s*\$([\d.]+)'),
            'select_rules': re.compile(r'select rules\s*(.+)'),
            'category_rule': re.compile(r'^-\s*Select rules\s+(.+?)\s+applies to all'),
            'rule_option': re.compile(r'^\s*(.+?)\s*\(Rule:\s*(.+?)\)(:)?'),
            'rule_item': re.compile(r'^(\s*)-\s*(.+?)\s*-\s*\$([\d.]+)\s*-?\s*(.*)'),
            'standard_item': re.compile(r'^-\s*(.*?):\s*\$(\d+\.\d{2})')
        }

    def parse_menu_file(self, file_path):
        """Parse menu file (your existing implementation)"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        item_rules = {}
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Category detection
            cat_match = self.patterns['category'].match(line)
            if cat_match:
                self.current_category = {
                    "name": cat_match.group(1).strip(),
                    "items": []
                }
                self.categories.append(self.current_category)
                i += 1
                continue
            
            # Category end detection
            if self.patterns['category_end'].match(line):
                self.current_category = None
                i += 1
                continue
            
            if self.current_category:
                # Category-wide rule directive
                category_rule_match = self.patterns['category_rule'].match(line)
                if category_rule_match:
                    rule_names = [rule.strip() for rule in category_rule_match.group(1).split(',')]
                    for item in self.current_category['items']:
                        if item['name'] not in item_rules:
                            item_rules[item['name']] = []
                        item_rules[item['name']].extend(rule_names)
                    i += 1
                    continue
                
                # Standard item detection
                item_match = self.patterns['standard_item'].match(line)
                if item_match:
                    item_name = item_match.group(1).strip()
                    price = float(item_match.group(2))
                    description = ""
                    rules = []
                    
                    # Check for rules in the item line
                    select_rules_match = self.patterns['select_rules'].search(line)
                    if select_rules_match:
                        rules = [rule.strip() for rule in select_rules_match.group(1).split(',')]
                    
                    # Get description from next line if it exists
                    if i + 1 < len(lines) and lines[i+1].startswith(" "):
                        description = lines[i+1].strip()
                        i += 2
                    else:
                        i += 1
                    
                    self.current_category['items'].append({
                        'name': item_name,
                        'price': price,
                        'description': description,
                        'selected_rules': rules
                    })
            else:
                i += 1
        
        # Add rules to items
        for category in self.categories:
            for item in category['items']:
                if item['name'] in item_rules:
                    item['selected_rules'] = item_rules[item['name']]

    def parse_rules_file(self, file_path):
        """Parse rules file (your existing implementation)"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Rule begin detection
            rule_begin_match = self.patterns['rule_begin'].match(line)
            if rule_begin_match:
                rule_name = rule_begin_match.group(1).strip()
                self.current_rule = {
                    "name": rule_name,
                    "options": []
                }
                self.rules.append(self.current_rule)
                i += 1
                continue
            
            # Rule end detection
            if self.patterns['rule_end'].match(line):
                self.current_rule = None
                i += 1
                continue
            
            # Option detection within a rule
            rule_option_match = self.patterns['rule_option'].match(line)
            if rule_option_match and self.current_rule:
                option_name = rule_option_match.group(1).strip()
                constraint_text = rule_option_match.group(2).strip()
                constraints = self._parse_constraints(constraint_text)
                
                current_option = {
                    "name": option_name,
                    "constraints": constraints,
                    "items": []
                }
                self.current_rule["options"].append(current_option)
                
                j = i + 1
                while j < len(lines):
                    item_line = lines[j].strip()
                    if not item_line or self.patterns['rule_option'].match(item_line) or self.patterns['rule_end'].match(item_line):
                        break
                    
                    rule_item_match = self.patterns['rule_item'].match(item_line)
                    if rule_item_match:
                        item_name = rule_item_match.group(2).strip()
                        price = float(rule_item_match.group(3))
                        description = rule_item_match.group(4)
                        if not description:
                            description = ""
                        
                        current_option["items"].append({
                            "name": item_name,
                            "price": price,
                            "description": description
                        })
                    j += 1
                
                i = j
                continue
            
            i += 1

    def _parse_constraints(self, text):
        """Parse rule constraints from text"""
        constraints = {'min': 0, 'max': None}
        text = text.lower()
        
        if 'select 1' in text and 'to' not in text:
            constraints.update({'min': 1, 'max': 1})
        elif 'up to' in text:
            match = re.search(r'up to (\d+)', text)
            if match:
                constraints['max'] = int(match.group(1))
        elif 'select' in text:
            match = re.search(r'select (\d+) to (\d+)', text)
            if match:
                constraints['min'] = int(match.group(1))
                constraints['max'] = int(match.group(2))
            else:
                match = re.search(r'select up to (\d+)', text)
                if match:
                    constraints['max'] = int(match.group(1))
        
        return constraints

# Enhanced Universal Menu Indexer
class UniversalMenuIndexer:
    def __init__(self, db_path: str = "universal_chroma_database"):
        self.client = PersistentClient(path=db_path)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self._initialize_collections()
        self.available_sectors = self._discover_sectors()

    def _initialize_collections(self):
        """Initialize all collections"""
        self.categories_col = self._get_or_create_collection("universal_categories")
        self.items_col = self._get_or_create_collection("universal_items")
        self.rules_col = self._get_or_create_collection("universal_rules")
        self.rule_options_col = self._get_or_create_collection("universal_rule_options")
        self.rule_items_col = self._get_or_create_collection("universal_rule_items")

    def _get_or_create_collection(self, name):
        """Get or create a collection with the given name"""
        try:
            return self.client.get_collection(name, self.embedder)
        except Exception:
            return self.client.create_collection(name, embedding_function=self.embedder)

    def _discover_sectors(self) -> List[str]:
        """Discover available sectors from sectors/ directory"""
        sectors = []
        sectors_dir = "sectors"
        
        if os.path.exists(sectors_dir):
            for item in os.listdir(sectors_dir):
                sector_path = os.path.join(sectors_dir, item)
                if os.path.isdir(sector_path):
                    # Check if it has required files
                    required_files = ["prompt2.txt", "rules.txt"]
                    if all(os.path.exists(os.path.join(sector_path, f)) for f in required_files):
                        sectors.append(item)
                        
        return sorted(sectors)

    def _get_sector_keywords(self, sector_name: str) -> List[str]:
        """Generic sector keywords - easily extensible for any new sector"""
        sector_keywords_map = {
            "food_delivery": [
                # Food items and meals
                "omelet", "omelette", "bagel", "sandwich", "salad", "burger", "pizza", "pasta", 
                "breakfast", "lunch", "dinner", "meal", "coffee", "juice", "smoothie",
                # Food actions and context
                "food", "order", "eat", "hungry", "delivery", "pickup", "restaurant", "deli",
                "menu", "cook", "fresh", "hot", "cold", "grilled", "toasted"
            ],
            "healthcare": [
                "medical", "doctor", "health", "appointment", "clinic", "patient", "symptoms",
                "hospital", "nurse", "physician", "checkup", "prescription", "medicine", "sick"
            ],
            "auto_repair": [
                "car", "vehicle", "automotive", "repair", "mechanic", "maintenance", "brake",
                "engine", "oil", "tire", "battery", "transmission", "garage", "auto"
            ],
            "beauty_salon": [
                "beauty", "hair", "salon", "nails", "spa", "facial", "makeup", "cut", "style",
                "manicure", "pedicure", "color", "highlights", "massage", "treatment"
            ],
            "legal_services": [
                "legal", "lawyer", "attorney", "law", "consultation", "contract", "court",
                "lawsuit", "divorce", "will", "estate", "legal advice", "litigation"
            ],
            "financial_services": [
                "financial", "money", "investment", "banking", "insurance", "loan", "credit",
                "mortgage", "financial planning", "tax", "accounting", "wealth"
            ],
            "real_estate": [
                "real estate", "property", "house", "home", "buy", "sell", "rent", "apartment",
                "condo", "realtor", "listing", "mortgage", "property management"
            ],
            "fitness_gym": [
                "fitness", "gym", "workout", "exercise", "training", "personal trainer",
                "muscle", "cardio", "weight", "yoga", "pilates", "membership"
            ],
            "photography": [
                "photography", "photo", "camera", "portrait", "wedding", "shoot", "pictures",
                "photographer", "album", "editing", "studio", "event photography"
            ],
            "pet_services": [
                "pet", "animal", "dog", "cat", "veterinary", "grooming", "vet", "puppy",
                "kitten", "pet care", "boarding", "training", "vaccination"
            ],
            "transportation": [
                "transport", "ride", "taxi", "travel", "airport", "uber", "lyft", "bus",
                "train", "car service", "shuttle", "transportation"
            ],
            "travel_hotel": [
                "travel", "hotel", "vacation", "booking", "accommodation", "flight", "trip",
                "resort", "tourism", "reservation", "hospitality", "lodge"
            ],
            "home_services": [
                "home", "repair", "maintenance", "plumbing", "electrical", "cleaning",
                "handyman", "renovation", "installation", "fix", "house repair"
            ],
            "education_tutoring": [
                "education", "tutoring", "learning", "teaching", "academic", "study",
                "tutor", "lesson", "homework", "test prep", "school", "learning"
            ],
            "insurance": [
                "insurance", "policy", "coverage", "protection", "claim", "premium",
                "deductible", "liability", "auto insurance", "health insurance"
            ],
            "event_planning": [
                "event", "party", "wedding", "planning", "celebration", "catering",
                "birthday", "anniversary", "corporate event", "venue", "planner"
            ],
            "moving_services": [
                "moving", "relocation", "packing", "storage", "furniture", "movers",
                "move", "relocate", "boxes", "moving company", "transport"
            ],
            "it_services": [
                "computer", "IT", "technology", "software", "repair", "virus", "laptop",
                "tech support", "network", "hardware", "IT support", "technical"
            ],
            "laundry_services": [
                "laundry", "cleaning", "wash", "dry cleaning", "clothes", "stain",
                "laundromat", "wash and fold", "garment", "fabric care"
            ]
        }
        
        return sector_keywords_map.get(sector_name, [sector_name.replace('_', ' ')])

    
    def _get_business_domain(self, sector_name: str) -> str:
        """Get business domain category for a sector"""
        domain_map = {
            "healthcare": "medical",
            "auto_repair": "automotive", 
            "beauty_salon": "personal_care",
            "legal_services": "professional_services",
            "financial_services": "financial",
            "real_estate": "real_estate",
            "fitness_gym": "health_fitness",
            "photography": "creative_services",
            "pet_services": "pet_care",
            "transportation": "transportation",
            "travel_hotel": "hospitality",
            "home_services": "home_maintenance",
            "education_tutoring": "education",
            "insurance": "financial",
            "event_planning": "event_services",
            "moving_services": "logistics",
            "it_services": "technology",
            "laundry_services": "personal_care",
            "food_delivery":"food_ordering"
        }
        
        return domain_map.get(sector_name, "general_services")

    def validate_option_metadata(self, metadata):
        """Ensure all metadata values are valid types for ChromaDB"""
        for key, value in metadata.items():
            if value is None:
                metadata[key] = -1 if key == "max" else ""
            elif not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)

    def clear_all_collections(self):
        """Clear all existing collections"""
        collections = ["universal_categories", "universal_items", "universal_rules", 
                      "universal_rule_options", "universal_rule_items"]
        
        for collection_name in collections:
            try:
                self.client.delete_collection(collection_name)
                print(f"[DEBUG] Deleted collection: {collection_name}")
            except Exception:
                pass
        
        self._initialize_collections()

    def index_all_sectors(self) -> Tuple[int, List[str]]:
        """Index all available sectors in one database"""
        print(f"🚀 Starting universal indexing for {len(self.available_sectors)} sectors...")
        
        self.clear_all_collections()
        
        total_indexed = 0
        failed_sectors = []
        
        for sector_name in self.available_sectors:
            try:
                print(f"\n🔄 Processing sector: {sector_name}")
                self.index_single_sector(sector_name)
                total_indexed += 1
                print(f"✅ Completed indexing: {sector_name}")
                
            except Exception as e:
                print(f"❌ Failed to index {sector_name}: {str(e)}")
                failed_sectors.append(sector_name)
        
        print(f"\n📊 INDEXING SUMMARY:")
        print(f"✅ Successfully indexed: {total_indexed} sectors")
        if failed_sectors:
            print(f"❌ Failed sectors: {failed_sectors}")
        
        self._print_database_stats()
        
        return total_indexed, failed_sectors

    def index_sectors(self, sector_names: List[str]) -> Tuple[int, List[str]]:
        """Index specific sectors"""
        print(f"🎯 Indexing specific sectors: {', '.join(sector_names)}")
        
        # Validate sector names
        invalid_sectors = [s for s in sector_names if s not in self.available_sectors]
        if invalid_sectors:
            print(f"❌ Invalid sectors: {invalid_sectors}")
            print(f"✅ Available sectors: {', '.join(self.available_sectors)}")
            return 0, invalid_sectors
        
        total_indexed = 0
        failed_sectors = []
        
        for sector_name in sector_names:
            try:
                print(f"\n🔄 Processing sector: {sector_name}")
                self.index_single_sector(sector_name)
                total_indexed += 1
                print(f"✅ Completed indexing: {sector_name}")
                
            except Exception as e:
                print(f"❌ Failed to index {sector_name}: {str(e)}")
                failed_sectors.append(sector_name)
        
        print(f"\n📊 INDEXING SUMMARY:")
        print(f"✅ Successfully indexed: {total_indexed}/{len(sector_names)} sectors")
        if failed_sectors:
            print(f"❌ Failed sectors: {failed_sectors}")
        
        return total_indexed, failed_sectors

    def index_single_sector(self, sector_name: str):
        """Index a single sector with sector metadata"""
        sector_path = f"sectors/{sector_name}"
        
        if not os.path.exists(sector_path):
            raise FileNotFoundError(f"Sector directory not found: {sector_path}")
        
        # Create parser for this sector
        menu_parser = MenuParser()
        
        # Parse sector files
        menu_file = f"{sector_path}/prompt2.txt"
        rules_file = f"{sector_path}/rules.txt"
        
        if not os.path.exists(menu_file):
            raise FileNotFoundError(f"Menu file not found: {menu_file}")
        
        if not os.path.exists(rules_file):
            raise FileNotFoundError(f"Rules file not found: {rules_file}")
        
        print(f"[DEBUG] Parsing menu file: {menu_file}")
        menu_parser.parse_menu_file(menu_file)
        
        print(f"[DEBUG] Parsing rules file: {rules_file}")
        menu_parser.parse_rules_file(rules_file)
        
        # Index with sector metadata
        self.index_menu_and_rules_with_sector(menu_parser, sector_name)
        
        print(f"[DEBUG] Completed indexing for sector: {sector_name}")

    def index_menu_and_rules_with_sector(self, menu_parser, sector_name: str):
        """Enhanced version of index_menu_and_rules with sector metadata"""
        print(f"[DEBUG] Indexing categories and items for sector: {sector_name}")
        
        # Add sector keywords for better search
        sector_keywords = self._get_sector_keywords(sector_name)
        business_domain = self._get_business_domain(sector_name)
        
        # Index rules first
        for rule in menu_parser.rules:
            rule_id = f"{sector_name}_rule_{rule['name']}"
            self.rules_col.add(
                documents=[f"{rule['name']} {' '.join(sector_keywords)}"],
                metadatas=[{
                    "name": rule['name'], 
                    "sector": sector_name,
                    "business_domain": business_domain
                }],
                ids=[rule_id]
            )
            
            # Index options for this rule
            for option in rule.get('options', []):
                option_id = f"{sector_name}_option_{rule['name']}_{option['name']}"
                option_metadata = {
                    "rule": rule['name'],
                    "name": option['name'],
                    "sector": sector_name,
                    "business_domain": business_domain,
                    "min": option['constraints'].get('min', 0),
                    "max": option['constraints'].get('max', -1)
                }
                
                self.validate_option_metadata(option_metadata)
                
                self.rule_options_col.add(
                    documents=[f"{option['name']} {' '.join(sector_keywords)}"],
                    metadatas=[option_metadata],
                    ids=[option_id]
                )
                
                # Index items for this option
                for item in option.get('items', []):
                    item_id = f"{sector_name}_rule_item_{rule['name']}_{option['name']}_{item['name']}"
                    item_metadata = {
                        "rule": rule['name'],
                        "option": option['name'],
                        "price": item['price'],
                        "item": item['name'],
                        "description": item.get("description", ""),
                        "sector": sector_name,
                        "business_domain": business_domain
                    }
                    
                    self.rule_items_col.add(
                        documents=[f"{item['name']} {item.get('description', '')} {' '.join(sector_keywords)}"],
                        metadatas=[item_metadata],
                        ids=[item_id]
                    )
        
        # Index categories and items
        for category in menu_parser.categories:
            category_id = f"{sector_name}_cat_{category['name']}"
            category_metadata = {
                'name': category['name'],
                'sector': sector_name,
                'business_domain': business_domain
            }
            
            if 'base_price' in category:
                category_metadata['base_price'] = category['base_price']
            
            if 'selected_rules' in category and category['selected_rules']:
                category_metadata['selected_rules'] = json.dumps(category['selected_rules'])
            
            self.categories_col.add(
                documents=[f"{category['name']} {' '.join(sector_keywords)}"],
                metadatas=[category_metadata],
                ids=[category_id]
            )
            
            # Index items in this category
            for item in category.get('items', []):
                self._index_item_with_sector(item, category, sector_name, sector_keywords, business_domain)

    def _index_item_with_sector(self, item, category, sector_name: str, sector_keywords: List[str], business_domain: str):
        """Index item with sector metadata"""
        item_id = f"{sector_name}_item_{category['name']}_{item['name']}"
        
        metadata = {
            'name': item['name'],
            'category': category['name'],
            'price': item['price'],
            'base_price': item['price'],
            'description': item.get('description', ''),
            'sector': sector_name,
            'business_domain': business_domain
        }
        
        if 'selected_rules' in item and item['selected_rules']:
            metadata['selected_rules'] = json.dumps(item['selected_rules'])
        
        # Enhanced document text with sector context
        document_text = f"{item['name']} {item.get('description', '')} {' '.join(sector_keywords)} {sector_name.replace('_', ' ')}"
        
        self.items_col.add(
            documents=[document_text],
            metadatas=[metadata],
            ids=[item_id]
        )

    def query_by_sector(self, query_text: str, sector_names: List[str] = None, n_results: int = 10):
        """Query the database with optional sector filtering"""
        if sector_names:
            where_clause = {"sector": {"$in": sector_names}}
        else:
            where_clause = None
        
        try:
            results = self.items_col.query(
                query_texts=[query_text],
                where=where_clause,
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            return self._format_query_results(results)
            
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return []

    def _format_query_results(self, results):
        """Format query results for better readability"""
        formatted_results = []
        
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                formatted_results.append({
                    "document": doc,
                    "sector": metadata.get("sector", "unknown"),
                    "name": metadata.get("name", ""),
                    "category": metadata.get("category", ""),
                    "price": metadata.get("price", 0),
                    "business_domain": metadata.get("business_domain", ""),
                    "relevance_score": 1 - distance,
                    "metadata": metadata
                })
        
        return formatted_results

    def list_available_sectors(self):
        """List all available sectors"""
        print(f"\n📋 AVAILABLE SECTORS ({len(self.available_sectors)}):")
        print("=" * 50)
        
        for i, sector in enumerate(self.available_sectors, 1):
            sector_path = f"sectors/{sector}"
            files_status = []
            
            # Check file status
            for filename in ["prompt.txt", "prompt2.txt", "rules.txt", "prompt3.txt"]:
                file_path = os.path.join(sector_path, filename)
                if os.path.exists(file_path):
                    files_status.append("✅")
                else:
                    files_status.append("❌")
            
            print(f"{i:2d}. {sector:20} | Files: {' '.join(files_status)} | Domain: {self._get_business_domain(sector)}")

    def _print_database_stats(self):
        """Print database statistics"""
        print(f"\n📊 DATABASE STATISTICS:")
        print("=" * 40)
        
        collections_info = {
            "categories": self.categories_col,
            "items": self.items_col,
            "rules": self.rules_col,
            "rule_options": self.rule_options_col,
            "rule_items": self.rule_items_col
        }
        
        total_documents = 0
        for name, collection in collections_info.items():
            try:
                count = collection.count()
                total_documents += count
                print(f"   {name:15}: {count:6,} documents")
            except Exception as e:
                print(f"   {name:15}: Error - {e}")
        
        print(f"   {'='*25}")
        print(f"   {'Total':15}: {total_documents:6,} documents")
        print(f"   {'Sectors':15}: {len(self.available_sectors):6,} sectors")

# Multi-Intent Detection System
class MultiIntentDetector:
    def __init__(self, indexer: UniversalMenuIndexer):
        self.indexer = indexer
        self.sector_keywords = {
            "food_delivery": [
            # Core food terms - HIGH PRIORITY
            "omelet", "omelette", "eggs", "food", "order", "breakfast", "lunch", "dinner",
            "bagel", "sandwich", "salad", "burger", "pizza", "pasta", "coffee",
            # Action terms
            "eat", "hungry", "meal", "delivery", "pickup", "restaurant", "deli",
            # Menu items from your actual menu
            "bagel special", "build your own breakfast", "french toast", "burrito",
            "wrap", "club", "soup", "appetizer", "dessert", "beverage",
            # Preparation terms
            "toasted", "grilled", "fresh", "hot", "cold"
            ],
            "healthcare": ["doctor", "medical", "appointment", "sick", "health", "clinic", "symptoms"],
            "auto_repair": ["car", "vehicle", "brake", "engine", "repair", "mechanic", "oil change"],
            "beauty_salon": ["hair", "nails", "facial", "makeup", "salon", "spa", "haircut"],
            "legal_services": ["lawyer", "attorney", "legal", "contract", "court", "law"],
            "financial_services": ["financial", "money", "investment", "banking", "insurance", "loan"],
            "real_estate": ["house", "home", "property", "buy", "sell", "rent", "real estate"],
            "fitness_gym": ["gym", "workout", "exercise", "fitness", "training", "personal trainer"],
            "photography": ["photo", "camera", "portrait", "wedding", "shoot", "photography"],
            "pet_services": ["pet", "dog", "cat", "vet", "grooming", "animal", "veterinary"],
            "transportation": ["ride", "taxi", "uber", "transport", "airport", "travel"],
            "travel_hotel": ["hotel", "vacation", "booking", "flight", "travel", "accommodation"],
            "home_services": ["plumbing", "electrical", "repair", "home", "maintenance", "cleaning"],
            "education_tutoring": ["tutor", "education", "learning", "study", "teaching", "academic"],
            "insurance": ["insurance", "policy", "coverage", "claim", "protection", "premium"],
            "event_planning": ["event", "party", "wedding", "planning", "celebration", "catering"],
            "moving_services": ["moving", "relocation", "packing", "movers", "furniture", "storage"],
            "it_services": ["computer", "laptop", "virus", "software", "tech", "IT", "repair"],
            "laundry_services": ["laundry", "wash", "dry cleaning", "clothes", "stain", "cleaning"]
        }
        
        # Common connection words to handle multi-intent splitting
        self.connection_words = ["and", "then", "also", "plus", "as well as", "along with"]
        
        # Patterns for multi-intent detection
        self.multi_intent_patterns = [
            r'(.+?)\s+and\s+(.+)',
            r'(.+?)\s+then\s+(.+)',
            r'(.+?)\s+also\s+(.+)',
            r'(.+?)\s+plus\s+(.+)',
            r'(.+?)\s+as well as\s+(.+)',
            r'(.+?)\s+along with\s+(.+)'
        ]

    def detect_intents(self, user_query: str) -> Dict[str, Any]:
        """Main intent detection method"""
        print(f"🔍 Analyzing query: '{user_query}'")
        
        # Step 1: Check for multi-intent patterns
        intents = self._split_multi_intents(user_query)
        
        if len(intents) > 1:
            print(f"🎯 Detected multiple intents: {len(intents)}")
            return self._process_multi_intents(intents, user_query)
        else:
            print(f"🎯 Detected single intent")
            return self._process_single_intent(user_query)

    def _split_multi_intents(self, query: str) -> List[str]:
        """Split query into multiple intents"""
        intents = []
        
        # Try each pattern to split the query
        for pattern in self.multi_intent_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Split into parts
                parts = [match.group(1).strip(), match.group(2).strip()]
                
                # Further split the second part if it contains more connectors
                for part in parts:
                    if any(word in part.lower() for word in self.connection_words):
                        # Recursively split
                        sub_intents = self._split_multi_intents(part)
                        intents.extend(sub_intents)
                    else:
                        intents.append(part)
                
                return intents
        
        # No multi-intent pattern found, return original query
        return [query]

   
    def _process_single_intent(self, query: str) -> Dict[str, Any]:
        """Fixed single intent processing with proper confidence calculation"""
        print(f"[DEBUG] Processing single intent for: '{query}'")
        
        # Method 1: Vector database search across ALL sectors
        print(f"[DEBUG] Querying vector database across all sectors...")
        vector_results = self.indexer.query_by_sector(query, n_results=20)
        print(f"[DEBUG] Vector results count: {len(vector_results)}")
        
        # Debug top vector results
        for i, result in enumerate(vector_results[:5]):
            print(f"[DEBUG] Vector #{i+1}: {result['sector']} - {result['name']} (score: {result['relevance_score']:.3f})")
        
        # Method 2: Generic keyword-based scoring
        print(f"[DEBUG] Calculating keyword scores...")
        keyword_scores = self._calculate_keyword_scores(query)
        
        # Method 3: FIXED confidence calculation
        sector_confidence = {}
        
        # FIXED: Score sectors from vector results using POSITIVE relevance scores
        print(f"[DEBUG] Calculating vector confidence scores...")
        for result in vector_results:
            sector = result["sector"]
            # Use relevance_score (0-1, higher = better) instead of distance
            relevance = result["relevance_score"]  # This should be positive
            
            if sector not in sector_confidence:
                sector_confidence[sector] = 0
            
            # Add positive relevance scores
            if relevance > 0:  # Only add positive relevance
                sector_confidence[sector] += relevance
                print(f"[DEBUG] Added {relevance:.3f} to {sector} (vector)")
        
        # Add keyword scores with appropriate weighting
        print(f"[DEBUG] Adding keyword scores...")
        for sector, score in keyword_scores.items():
            if sector not in sector_confidence:
                sector_confidence[sector] = 0
            
            # Add keyword scores (already positive)
            weighted_keyword_score = score * 2.0  # 2x weight for exact keyword matches
            sector_confidence[sector] += weighted_keyword_score
            print(f"[DEBUG] Added {weighted_keyword_score:.3f} to {sector} (keywords)")
        
        print(f"[DEBUG] Final sector confidence: {sector_confidence}")
        
        # Sort by confidence (highest first)
        sorted_sectors = sorted(sector_confidence.items(), key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] Sorted sectors: {sorted_sectors}")
        
        # Determine final result
        if sorted_sectors:
            primary_sector = sorted_sectors[0][0]
            confidence = sorted_sectors[0][1]
        else:
            primary_sector = "unknown"
            confidence = 0.0
        
        print(f"[DEBUG] Final detection: {primary_sector} with confidence {confidence:.2f}")
        
        return {
            "intent_type": "single",
            "query": query,
            "primary_sector": primary_sector,
            "confidence": confidence,
            "alternative_sectors": [s[0] for s in sorted_sectors[1:3]],
            "vector_results": vector_results[:5],
            "detected_services": [r["name"] for r in vector_results[:3]],
            "debug_info": {
                "keyword_scores": keyword_scores,
                "vector_count": len(vector_results),
                "final_scores": dict(sorted_sectors),
                "top_vector_sectors": [r["sector"] for r in vector_results[:5]]
            }
        }


    def _process_multi_intents(self, intents: List[str], original_query: str) -> Dict[str, Any]:
        """Process multiple intents"""
        intent_results = []
        all_sectors = set()
        
        for i, intent in enumerate(intents):
            print(f"  📍 Processing intent {i+1}: '{intent}'")
            result = self._process_single_intent(intent)
            intent_results.append(result)
            all_sectors.add(result["primary_sector"])
        
        return {
            "intent_type": "multi",
            "original_query": original_query,
            "split_intents": intents,
            "intent_results": intent_results,
            "involved_sectors": list(all_sectors),
            "total_intents": len(intents),
            "complexity": "high" if len(all_sectors) > 2 else "medium"
        }

    def _calculate_keyword_scores(self, query: str) -> Dict[str, float]:
        """Completely generic keyword scoring - no hardcoded sector logic"""
        query_lower = query.lower()
        scores = {}
        
        print(f"[DEBUG] Calculating generic keyword scores for: '{query}'")
        
        # Generic scoring for ALL sectors equally
        for sector, keywords in self.sector_keywords.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in query_lower:
                    # Exact word boundary match gets higher score
                    if f" {keyword} " in f" {query_lower} ":
                        score += 2.0
                        matched_keywords.append(f"{keyword}(exact)")
                    # Partial match gets lower score
                    elif keyword in query_lower:
                        score += 1.0
                        matched_keywords.append(f"{keyword}(partial)")
            
            if score > 0:
                scores[sector] = score
                print(f"[DEBUG] {sector}: {score:.1f} points from {matched_keywords}")
        
        print(f"[DEBUG] Final keyword scores: {scores}")
        return scores

# Complete Universal Service Bot Application
class UniversalServiceBot:
    def __init__(self, db_path: str = "universal_chroma_database"):
        self.indexer = UniversalMenuIndexer(db_path)
        self.intent_detector = MultiIntentDetector(self.indexer)

    def process_query(self, query: str, filter_sectors: List[str] = None) -> Dict[str, Any]:
        """Process a user query with intent detection"""
        # Detect intents
        intent_analysis = self.intent_detector.detect_intents(query)
        
        # Get relevant services
        if intent_analysis["intent_type"] == "single":
            primary_sector = intent_analysis["primary_sector"]
            sectors_to_search = [primary_sector] if primary_sector != "unknown" else None
            
            if filter_sectors:
                sectors_to_search = filter_sectors
            
            services = self.indexer.query_by_sector(query, sectors_to_search, n_results=10)
            
            return {
                **intent_analysis,
                "recommended_services": services[:5],
                "action": "single_sector_response"
            }
        
        else:  # multi-intent
            return {
                **intent_analysis,
                "action": "multi_sector_coordination"
            }

    def run_interactive_mode(self):
        """Run interactive query mode"""
        print("🤖 Universal Service Bot - Interactive Mode")
        print("=" * 50)
        print("Type your service requests (e.g., 'I need a haircut and food delivery')")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                query = input("🗣️  You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Process the query
                result = self.process_query(query)
                
                # Display results
                self._display_query_results(result)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _display_query_results(self, result: Dict[str, Any]):
        """Display query results in a user-friendly format"""
        if result["intent_type"] == "single":
            print(f"\n🎯 Detected Service: {result['primary_sector'].replace('_', ' ').title()}")
            print(f"🔍 Confidence: {result['confidence']:.2f}")
            
            if result.get("recommended_services"):
                print(f"\n📋 Available Services:")
                for i, service in enumerate(result["recommended_services"], 1):
                    print(f"  {i}. {service['name']} - ${service['price']}")
                    if service.get('category'):
                        print(f"     Category: {service['category']}")
            
            if result.get("alternative_sectors"):
                print(f"\n🔄 Alternative Sectors: {', '.join(result['alternative_sectors'])}")
        
        else:  # multi-intent
            print(f"\n🎯 Detected Multiple Requests ({result['total_intents']} intents)")
            print(f"🏢 Involved Sectors: {', '.join([s.replace('_', ' ').title() for s in result['involved_sectors']])}")
            
            for i, intent_result in enumerate(result["intent_results"], 1):
                print(f"\n  📍 Request {i}: {result['split_intents'][i-1]}")
                print(f"      ➜ Sector: {intent_result['primary_sector'].replace('_', ' ').title()}")
                print(f"      ➜ Confidence: {intent_result['confidence']:.2f}")

def create_argument_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Universal Service Bot - Multi-sector intent detection and service routing',
        epilog="""
Examples:
  # Index all sectors
  python universal_service_bot.py --index-all
  
  # Index specific sectors
  python universal_service_bot.py --index-sectors healthcare auto_repair
  
  # Single intent detection
  python universal_service_bot.py --query "I need a doctor appointment"
  
  # Multi-intent detection  
  python universal_service_bot.py --query "I need food delivery and a haircut" --multi-intent
  
  # Interactive mode
  python universal_service_bot.py --interactive
  
  # List available sectors
  python universal_service_bot.py --list-sectors
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    
    # Indexing actions
    action_group.add_argument(
        '--index-all',
        action='store_true',
        help='Index all available sectors into the universal database'
    )
    
    action_group.add_argument(
        '--index-sectors',
        nargs='+',
        metavar='SECTOR',
        help='Index specific sectors (space-separated list)'
    )
    
    # Query actions
    action_group.add_argument(
        '--query', '-q',
        metavar='TEXT',
        help='Process a single query for intent detection'
    )
    
    action_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode for multiple queries'
    )
    
    # Information actions
    action_group.add_argument(
        '--list-sectors', '-l',
        action='store_true',
        help='List all available sectors'
    )
    
    action_group.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    
    # Optional arguments
    parser.add_argument(
        '--multi-intent',
        action='store_true',
        help='Enable multi-intent detection (use with --query)'
    )
    
    parser.add_argument(
        '--filter-sectors',
        nargs='+',
        metavar='SECTOR',
        help='Filter results by specific sectors (use with --query)'
    )
    
    parser.add_argument(
        '--results', '-n',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    
    parser.add_argument(
        '--db-path',
        default='universal_chroma_database',
        help='Path to ChromaDB database (default: universal_chroma_database)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser

def main():
    """Main function with command-line interface"""
    parser = create_argument_parser()
    
    try:
        args = parser.parse_args()
    except SystemExit:
        return
    
    # Initialize the universal service bot
    print("🤖 Universal Service Bot - Multi-Sector Intent Detection")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    bot = UniversalServiceBot(db_path=args.db_path)
    
    if args.verbose:
        print(f"📂 Database path: {args.db_path}")
        print(f"🔍 Discovered {len(bot.indexer.available_sectors)} sectors")
    
    # Execute requested action
    if args.index_all:
        # Index all sectors
        total, failed = bot.indexer.index_all_sectors()
        if total > 0:
            print(f"\n🎉 Successfully indexed {total} sectors!")
        if failed:
            print(f"⚠️  Failed to index: {failed}")
            sys.exit(1)
    
    elif args.index_sectors:
        # Index specific sectors
        total, failed = bot.indexer.index_sectors(args.index_sectors)
        if total > 0:
            print(f"\n🎉 Successfully indexed {total} sectors!")
        if failed:
            print(f"⚠️  Failed to index: {failed}")
            sys.exit(1)
    
    elif args.query:
        # Process query with intent detection
        print(f"🔍 Processing query: '{args.query}'")
        
        if args.multi_intent:
            print("🎯 Multi-intent detection enabled")
        
        result = bot.process_query(
            query=args.query,
            filter_sectors=args.filter_sectors
        )
        
        # Display results
        print(f"\n📊 INTENT ANALYSIS RESULTS:")
        print("-" * 50)
        bot._display_query_results(result)
        
        if args.verbose:
            print(f"\n🔧 Technical Details:")
            print(f"Intent Type: {result['intent_type']}")
            if result['intent_type'] == 'single':
                print(f"Primary Sector: {result['primary_sector']}")
                print(f"Confidence: {result['confidence']:.3f}")
            else:
                print(f"Split Intents: {result['split_intents']}")
                print(f"Involved Sectors: {result['involved_sectors']}")
    
    elif args.interactive:
        # Run interactive mode
        bot.run_interactive_mode()
    
    elif args.list_sectors:
        # List available sectors
        bot.indexer.list_available_sectors()
    
    elif args.stats:
        # Show database statistics
        bot.indexer._print_database_stats()

if __name__ == "__main__":
    main()

