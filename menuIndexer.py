#!/usr/bin/env python3

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import re
import json
import os
import argparse
import sys
from typing import List, Dict, Any

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
                # Check for category-wide rule directive
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
        """Parse the rules file to extract rule definitions, options, and rule items."""
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
        """Parse rule constraints from text."""
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
        """Get relevant keywords for a sector to improve search"""
        sector_keywords_map = {
            "healthcare": ["medical", "doctor", "health", "appointment", "clinic", "patient"],
            "auto_repair": ["car", "vehicle", "automotive", "repair", "mechanic", "maintenance"],
            "beauty_salon": ["beauty", "hair", "salon", "nails", "spa", "facial"],
            "legal_services": ["legal", "lawyer", "attorney", "law", "consultation"],
            "financial_services": ["financial", "money", "investment", "banking", "insurance"],
            "real_estate": ["real estate", "property", "house", "home", "buy", "sell"],
            "fitness_gym": ["fitness", "gym", "workout", "exercise", "training"],
            "photography": ["photography", "photo", "camera", "portrait", "wedding"],
            "pet_services": ["pet", "animal", "dog", "cat", "veterinary", "grooming"],
            "transportation": ["transport", "ride", "taxi", "travel", "airport"],
            "travel_hotel": ["travel", "hotel", "vacation", "booking", "accommodation"],
            "home_services": ["home", "repair", "maintenance", "plumbing", "electrical"],
            "education_tutoring": ["education", "tutoring", "learning", "teaching", "academic"],
            "insurance": ["insurance", "policy", "coverage", "protection", "claim"],
            "event_planning": ["event", "party", "wedding", "planning", "celebration"],
            "moving_services": ["moving", "relocation", "packing", "storage", "furniture"],
            "it_services": ["computer", "IT", "technology", "software", "repair"],
            "laundry_services": ["laundry", "cleaning", "wash", "dry cleaning", "clothes"]
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
            "laundry_services": "personal_care"
        }
        
        return domain_map.get(sector_name, "general_services")

    def validate_option_metadata(self, metadata):
        """Ensure all metadata values are valid types for ChromaDB."""
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
                pass  # Collection doesn't exist
        
        # Reinitialize collections
        self._initialize_collections()

    def index_all_sectors(self) -> tuple:
        """Index all available sectors in one database"""
        print(f"🚀 Starting universal indexing for {len(self.available_sectors)} sectors...")
        
        # Clear existing collections
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

    def index_sectors(self, sector_names: List[str]) -> tuple:
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

def create_argument_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Universal Menu Indexer - Index multiple service sectors in vector database',
        epilog="""
Examples:
  python menuIndexer.py --all                          # Index all sectors
  python menuIndexer.py --sectors healthcare auto_repair  # Index specific sectors
  python menuIndexer.py --list                         # List available sectors
  python menuIndexer.py --stats                        # Show database statistics
  python menuIndexer.py --query "doctor appointment"   # Query database
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    
    action_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Index all available sectors'
    )
    
    action_group.add_argument(
        '--sectors', '-s',
        nargs='+',
        metavar='SECTOR',
        help='Index specific sectors (space-separated list)'
    )
    
    action_group.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available sectors'
    )
    
    action_group.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    
    action_group.add_argument(
        '--query', '-q',
        metavar='TEXT',
        help='Query the database'
    )
    
    # Optional arguments
    parser.add_argument(
        '--db-path',
        default='universal_chroma_database',
        help='Path to ChromaDB database (default: universal_chroma_database)'
    )
    
    parser.add_argument(
        '--filter-sectors',
        nargs='+',
        metavar='SECTOR',
        help='Filter query results by specific sectors (use with --query)'
    )
    
    parser.add_argument(
        '--results', '-n',
        type=int,
        default=10,
        help='Number of query results to return (default: 10)'
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
    
    # Initialize indexer
    print("🤖 Universal Menu Indexer")
    print("=" * 50)
    
    indexer = UniversalMenuIndexer(db_path=args.db_path)
    
    if args.verbose:
        print(f"📂 Database path: {args.db_path}")
        print(f"🔍 Discovered {len(indexer.available_sectors)} sectors")
    
    # Execute requested action
    if args.all:
        # Index all sectors
        total, failed = indexer.index_all_sectors()
        if total > 0:
            print(f"\n🎉 Successfully indexed {total} sectors!")
        if failed:
            print(f"⚠️  Failed to index: {failed}")
            sys.exit(1)
    
    elif args.sectors:
        # Index specific sectors
        total, failed = indexer.index_sectors(args.sectors)
        if total > 0:
            print(f"\n🎉 Successfully indexed {total} sectors!")
        if failed:
            print(f"⚠️  Failed to index: {failed}")
            sys.exit(1)
    
    elif args.list:
        # List available sectors
        indexer.list_available_sectors()
    
    elif args.stats:
        # Show database statistics
        indexer._print_database_stats()
    
    elif args.query:
        # Query database
        print(f"🔍 Searching for: '{args.query}'")
        
        filter_sectors = args.filter_sectors if args.filter_sectors else None
        if filter_sectors:
            print(f"🎯 Filtering by sectors: {', '.join(filter_sectors)}")
        
        results = indexer.query_by_sector(
            query_text=args.query,
            sector_names=filter_sectors,
            n_results=args.results
        )
        
        if results:
            print(f"\n📋 Found {len(results)} results:")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                print(f"{i:2d}. {result['name']} ({result['sector']})")
                print(f"    Category: {result['category']} | Price: ${result['price']}")
                print(f"    Relevance: {result['relevance_score']:.3f} | Domain: {result['business_domain']}")
                if args.verbose:
                    print(f"    Document: {result['document'][:100]}...")
                print()
        else:
            print("❌ No results found")

if __name__ == "__main__":
    main()

