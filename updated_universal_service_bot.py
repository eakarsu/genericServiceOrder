#!/usr/bin/env python3

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangGraph and LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# ChromaDB imports
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

##########################
# UNIVERSAL DATABASE ACCESS
##########################

class UniversalDatabaseClient:
    """Direct access to universal ChromaDB without MenuIndexer/OrderProcessor dependencies"""
    
    def __init__(self, db_path: str = "universal_chroma_database"):
        self.client = PersistentClient(path=db_path)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self._initialize_collections()
        print(f"✅ Connected to universal database: {db_path}")
    
    def _initialize_collections(self):
        """Initialize connections to existing collections"""
        try:
            self.categories_col = self.client.get_collection("universal_categories", self.embedder)
            self.items_col = self.client.get_collection("universal_items", self.embedder)
            self.rules_col = self.client.get_collection("universal_rules", self.embedder)
            self.rule_options_col = self.client.get_collection("universal_rule_options", self.embedder)
            self.rule_items_col = self.client.get_collection("universal_rule_items", self.embedder)
            print("✅ All collections connected successfully")
        except Exception as e:
            print(f"❌ Error connecting to collections: {e}")
            raise RuntimeError("Universal database not available. Please ensure it's properly indexed.")
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and return statistics"""
        try:
            stats = {}
            collections = {
                "categories": self.categories_col,
                "items": self.items_col,
                "rules": self.rules_col,
                "rule_options": self.rule_options_col,
                "rule_items": self.rule_items_col
            }
            
            total_documents = 0
            for name, collection in collections.items():
                count = collection.count()
                stats[name] = count
                total_documents += count
            
            stats["total_documents"] = total_documents
            stats["healthy"] = total_documents > 0
            
            return stats
        except Exception as e:
            return {"healthy": False, "error": str(e)}

##########################
# SECTOR-SPECIFIC PROCESSOR (WITHOUT DEPENDENCIES)
##########################

class SectorSpecificProcessor:
    """Direct processor that queries universal database with sector filtering"""
    
    def __init__(self, db_client: UniversalDatabaseClient, sector_name: str):
        self.db_client = db_client
        self.sector_name = sector_name
        print(f"[DEBUG] Initialized SectorSpecificProcessor for: {sector_name}")
    
    def is_greeting(self, query: str) -> bool:
        """Check if query is a greeting"""
        query_lower = query.lower().strip()
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        return any(query_lower == greeting for greeting in greetings)
    
    def process_order(self, query: str) -> Dict[str, Any]:
        """Process order query with sector filtering"""
        print(f"[DEBUG] Processing order for sector '{self.sector_name}': {query}")
        
        if self.is_greeting(query):
            print(f"[DEBUG] Detected greeting: {query}")
            return {"status": "greeting"}
        
        # Use sector-filtered unified search
        unified_results = self.sector_filtered_unified_search(query)
        
        if not unified_results:
            print(f"[DEBUG] No results found for query in sector {self.sector_name}: {query}")
            return {"status": "no_results"}
        
        # Process results
        return self._process_sector_results(unified_results, query)
    
    def sector_filtered_unified_search(self, query: str) -> List[Dict[str, Any]]:
        """Unified search filtered by sector"""
        print(f"[DEBUG] Performing sector-filtered search for: {self.sector_name}")
        
        # Search all collections with sector filter
        where_clause = {"sector": self.sector_name}
        
        # Search categories for this sector
        category_results = self.db_client.categories_col.query(
            query_texts=[query],
            where=where_clause,
            n_results=10,
            include=["metadatas", "documents", "distances"]
        )
        
        # Search items for this sector
        items_results = self.db_client.items_col.query(
            query_texts=[query],
            where=where_clause,
            n_results=20,
            include=["metadatas", "documents", "distances"]
        )
        
        # Search rule options for this sector
        rule_options = self.db_client.rule_options_col.query(
            query_texts=[query],
            where=where_clause,
            n_results=10,
            include=["metadatas", "documents", "distances"]
        )
        
        # Search rule items for this sector
        rule_items = self.db_client.rule_items_col.query(
            query_texts=[query],
            where=where_clause,
            n_results=20,
            include=["metadatas", "documents", "distances"]
        )
        
        # Combine results
        unified_results = []
        
        # Add categories
        if category_results and category_results.get("ids") and len(category_results["ids"]) > 0:
            for i in range(len(category_results["ids"][0])):
                unified_results.append(self._create_result_item(category_results, i, "category"))
        
        # Add items
        if items_results and items_results.get("ids") and len(items_results["ids"]) > 0:
            for i in range(len(items_results["ids"][0])):
                unified_results.append(self._create_result_item(items_results, i, "item"))
        
        # Add rule options
        if rule_options and rule_options.get("ids") and len(rule_options["ids"]) > 0:
            for i in range(len(rule_options["ids"][0])):
                unified_results.append(self._create_result_item(rule_options, i, "rule_option"))
        
        # Add rule items
        if rule_items and rule_items.get("ids") and len(rule_items["ids"]) > 0:
            for i in range(len(rule_items["ids"][0])):
                unified_results.append(self._create_result_item(rule_items, i, "rule_item"))
        
        # Sort by score (lower distance = better match)
        unified_results.sort(key=lambda x: x.get("score", 1.0))
        
        print(f"[DEBUG] Found {len(unified_results)} results for sector {self.sector_name}")
        return unified_results
    
    def _create_result_item(self, results: Dict, index: int, item_type: str) -> Dict[str, Any]:
        """Create a standardized result item from database results"""
        doc = results["documents"][0][index]
        meta = results["metadatas"][0][index]
        distance = results["distances"][0][index]
        
        return {
            "type": item_type,
            "name": doc,
            "score": distance,
            "metadata": meta
        }
    
    def _process_sector_results(self, unified_results: List[Dict], query: str) -> Dict[str, Any]:
        """Process unified results with sector context"""
        top_matches = []
        result_status = "need_input"
        
        # Filter for exact matches (score < 0.9) in this sector
        for item_result in unified_results:
            if item_result["score"] < 0.9:
                meta = item_result["metadata"]
                
                # Verify this item belongs to our sector
                if meta.get("sector") == self.sector_name:
                    item = {
                        "item": item_result["name"],
                        "type": item_result["type"],
                        "ingredients": meta.get("ingredients", ""),
                        "price": meta.get("price", 0),
                        "category": meta.get("category", ""),
                        "sector": meta.get("sector", "")
                    }
                    
                    if "base_price" in meta:
                        item["base_price"] = meta["base_price"]
                    
                    if "selected_rules" in meta:
                        item["selected_rules"] = meta["selected_rules"]
                        result_status = "need_rule_selections"
                    
                    top_matches.append(item)
        
        # If no exact matches, try looser threshold
        if not top_matches:
            for item_result in unified_results:
                if item_result["score"] < 1.1:
                    meta = item_result["metadata"]
                    
                    # Verify this item belongs to our sector
                    if meta.get("sector") == self.sector_name:
                        item = {
                            "item": item_result["name"],
                            "type": item_result.get("type", "item"),
                            "ingredients": meta.get("ingredients", ""),
                            "price": meta.get("price", 0),
                            "category": meta.get("category", ""),
                            "sector": meta.get("sector", "")
                        }
                        
                        if "base_price" in meta:
                            item["base_price"] = meta["base_price"]
                        
                        if "selected_rules" in meta:
                            item["selected_rules"] = meta["selected_rules"]
                            result_status = "need_rule_selections"
                        
                        top_matches.append(item)
        
        result = {
            "status": result_status,
            "results": top_matches,
            "sector": self.sector_name
        }
        
        if top_matches:
            print(f"[DEBUG] Returning {len(top_matches)} matches for sector {self.sector_name}")
            return result
        else:
            print(f"[DEBUG] No matches found in sector {self.sector_name}")
            return {"status": "no_results", "sector": self.sector_name}

##########################
# ENHANCED SECTOR INTENT DETECTOR
##########################

##########################
# ENHANCED SECTOR INTENT DETECTOR (COMPLETE)
##########################

class EnhancedSectorIntentDetector:
    """Enhanced intent detection with vector database integration for better accuracy"""
    
    def __init__(self, db_client: UniversalDatabaseClient):
        self.db_client = db_client  # Store database client for vector searches
        self.sector_keywords = {
            "healthcare": ["doctor", "medical", "health", "appointment", "sick", "symptoms", "clinic", "hospital", "checkup", "physician", "nurse"],
            "food_delivery": ["food", "breakfast", "lunch", "dinner", "order", "omelet", "bagel", "sandwich", "hungry", "eat", "meal", "restaurant", "delivery"],
            "beauty_salon": ["hair", "nails", "facial", "makeup", "salon", "spa", "haircut", "manicure", "beauty", "makeover", "style", "color"],
            "legal_services": ["lawyer", "attorney", "legal", "contract", "court", "law", "divorce", "consultation", "litigation", "lawsuit"],
            "financial_services": ["financial", "money", "investment", "banking", "insurance", "loan", "planning", "credit", "mortgage"],
            "real_estate": ["house", "home", "property", "buy", "sell", "rent", "real estate", "apartment", "condo", "realtor"],
            "fitness_gym": ["gym", "workout", "exercise", "fitness", "training", "membership", "personal trainer", "cardio"],
            "photography": ["photo", "camera", "portrait", "wedding", "shoot", "photography", "pictures", "photographer"],
            "pet_services": ["pet", "dog", "cat", "vet", "grooming", "animal", "veterinary", "puppy", "kitten"],
            "transportation": ["ride", "taxi", "uber", "transport", "airport", "travel", "car service", "lyft"],
            "travel_hotel": ["hotel", "vacation", "booking", "flight", "travel", "accommodation", "trip", "resort"],
            "home_services": ["plumbing", "electrical", "repair", "home", "maintenance", "handyman", "renovation"],
            "education_tutoring": ["tutor", "education", "learning", "study", "teaching", "homework", "academic"],
            "insurance": ["insurance", "policy", "coverage", "claim", "protection", "premium", "deductible"],
            "event_planning": ["event", "party", "wedding", "planning", "celebration", "catering", "birthday"],
            "moving_services": ["moving", "relocation", "packing", "movers", "furniture", "move", "relocate"],
            "it_services": ["computer", "laptop", "virus", "software", "tech", "IT", "repair", "hardware", "network"],
            "laundry_services": ["laundry", "wash", "dry cleaning", "clothes", "stain", "cleaning", "garment"],
            "auto_repair": ["car", "vehicle", "automotive", "repair", "mechanic", "brake", "engine", "oil"]
        }

    def detect_sector_from_conversation(self, conversation_messages: List[str], max_messages: int = 3) -> str:
        """Detect sector using first few messages for better accuracy"""
        if not conversation_messages:
            return "food_delivery"  # Default sector
        
        # Skip generic greetings and combine meaningful content
        meaningful_content = self._extract_meaningful_content(conversation_messages, max_messages)
        
        if not meaningful_content:
            return "food_delivery"  # Default if no meaningful content
        
        # Detect from combined meaningful content
        detected_sector = self.detect_sector(meaningful_content)
        print(f"[DEBUG] Intent detection using: '{meaningful_content}' → {detected_sector}")
        
        return detected_sector
    
    def _extract_meaningful_content(self, messages: List[str], max_messages: int = 3) -> str:
        """Extract meaningful content from conversation, skipping greetings"""
        meaningful_parts = []
        
        for i, message in enumerate(messages[:max_messages]):
            message_clean = message.strip().lower()
            
            # Skip generic greetings
            if self._is_generic_greeting(message_clean):
                print(f"[DEBUG] Skipping generic greeting: '{message}'")
                continue
            
            # Add meaningful content
            meaningful_parts.append(message)
            print(f"[DEBUG] Added meaningful content: '{message}'")
        
        # Combine all meaningful parts
        combined_content = " ".join(meaningful_parts)
        return combined_content
    
    def _is_generic_greeting(self, message: str) -> bool:
        """Check if message is just a generic greeting"""
        generic_greetings = [
            "hello", "hi", "hey", "good morning", "good afternoon", 
            "good evening", "howdy", "what's up", "yo", "greetings"
        ]
        
        # Check if message is ONLY a greeting (with possible punctuation)
        clean_message = re.sub(r'[^\w\s]', '', message.lower()).strip()
        
        return clean_message in generic_greetings
    
    def detect_sector_with_confidence(self, user_input: str) -> Dict[str, Any]:
        """Enhanced detection with confidence scoring like in your successful tests"""
        print(f"[DEBUG] Detecting sector for: '{user_input}'")
        
        # Method 1: Vector database search across ALL sectors
        vector_results = self._query_all_sectors(user_input)
        
        # Method 2: Keyword-based scoring
        keyword_scores = self._calculate_keyword_scores(user_input)
        
        # Method 3: Combine results
        sector_confidence = {}
        
        # Score from vector results
        for result in vector_results:
            sector = result.get("sector")
            relevance = result.get("relevance_score", 0)
            if sector not in sector_confidence:
                sector_confidence[sector] = 0
            sector_confidence[sector] += relevance * 0.3
        
        # Score from keywords (higher weight)
        for sector, score in keyword_scores.items():
            if sector not in sector_confidence:
                sector_confidence[sector] = 0
            sector_confidence[sector] += score * 2.0  # Higher weight for keywords
        
        # Sort by confidence
        sorted_sectors = sorted(sector_confidence.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_sectors:
            primary_sector = sorted_sectors[0][0]
            confidence = sorted_sectors[0][1]
        else:
            primary_sector = "food_delivery"
            confidence = 0.0
        
        print(f"[DEBUG] Selected sector: {primary_sector} with confidence {confidence}")
        
        return {
            "primary_sector": primary_sector,
            "confidence": confidence,
            "alternative_sectors": [s[0] for s in sorted_sectors[1:3]],
            "keyword_scores": keyword_scores,
            "sector_confidence": dict(sorted_sectors)
        }
    
    def _query_all_sectors(self, query: str) -> List[Dict]:
        """Query vector database across all sectors"""
        try:
            results = self.db_client.items_col.query(
                query_texts=[query],
                n_results=20,
                include=["metadatas", "documents", "distances"]
            )
            
            formatted_results = []
            if results and results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    formatted_results.append({
                        "sector": metadata.get("sector"),
                        "relevance_score": 1 - distance,  # Convert distance to relevance
                        "name": metadata.get("name", ""),
                        "metadata": metadata
                    })
            
            return formatted_results
        except Exception as e:
            print(f"[DEBUG] Vector query error: {e}")
            return []
    
    def _calculate_keyword_scores(self, query: str) -> Dict[str, float]:
        """Enhanced keyword scoring with debug output like in your successful tests"""
        query_lower = query.lower()
        scores = {}
        
        print(f"[DEBUG] Calculating keyword scores for: '{query}'")
        
        for sector, keywords in self.sector_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in query_lower:
                    # Exact word boundary match gets higher score
                    if f" {keyword} " in f" {query_lower} ":
                        score += 2.0
                        matched_keywords.append(f"{keyword}(exact)")
                    else:
                        score += 1.0
                        matched_keywords.append(f"{keyword}(partial)")
            
            if score > 0:
                scores[sector] = score
                print(f"[DEBUG] {sector}: {score:.1f} points from {matched_keywords}")
        
        return scores
    
    def detect_sector(self, user_input: str) -> str:
        """Main detection method that returns just the sector name"""
        result = self.detect_sector_with_confidence(user_input)
        return result["primary_sector"]

##########################
# OPENROUTER INTEGRATION
##########################

class OpenRouterIntegration:
    """OpenRouter AI integration following orderChat.py pattern"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            print("❌ OPENROUTER_API_KEY environment variable not found")
            self.llm_local = None
            return
        
        try:
            self.llm_local = ChatOpenAI(
                model="openai/gpt-4o",
                openai_api_key=self.api_key,
                openai_api_base=self.api_url,
                default_headers={
                    "HTTP-Referer": "https://universal-service-bot.com",
                    "X-Title": "Universal Service Bot",
                    "Content-Type": "application/json"
                },
                max_retries=3,
                timeout=30
            )
            print(f"✅ OpenRouter client initialized with key: {self.api_key[:10]}...")
        except Exception as e:
            print(f"❌ Failed to initialize OpenRouter client: {e}")
            self.llm_local = None
    
    def call_openrouter_agent(self, sector: str, user_input: str, sector_prompt: str, chat_history: list = None) -> str:
        """Call OpenRouter AI following orderChat.py pattern"""
        if not self.llm_local:
            return f"I understand you're interested in {sector.replace('_', ' ')}. How can I help you today?"
        
        # Prepare messages
        messages = [
            SystemMessage(content=sector_prompt),
            *(chat_history or []),
            HumanMessage(content=user_input)
        ]
        
        try:
            print(f"🔄 Calling OpenRouter with key: {self.api_key[:10]}...")
            response = self.llm_local.invoke(messages)
            print(f"✅ OpenRouter response received: {len(response.content)} chars")
            return response.content
        except Exception as e:
            error_msg = str(e)
            print(f"❌ OpenRouter error: {error_msg}")
            
            if "401" in error_msg:
                print("🔑 Authentication failed - check your OpenRouter API key")
            elif "rate limit" in error_msg.lower():
                print("⏰ Rate limit exceeded")
            elif "insufficient funds" in error_msg.lower():
                print("💰 Insufficient OpenRouter credits")
            
            return f"I'm here to help with {sector.replace('_', ' ')}. Could you tell me more about what you need?"

##########################
# UNIVERSAL SERVICE BOT (MAIN CLASS)
##########################

class UniversalServiceBot:
    """Universal Service Bot that works directly with universal database"""
    
    
    def __init__(self, sectors_directory: str = "sectors", db_path: str = "universal_chroma_database"):
        self.sectors_directory = sectors_directory
        self.db_path = db_path
        
        # Initialize universal database client FIRST
        self.db_client = UniversalDatabaseClient(db_path)
        
        # Check database health
        db_stats = self.db_client.check_database_health()
        if not db_stats.get("healthy", False):
            raise RuntimeError(f"Universal database not healthy: {db_stats}")
        
        print(f"📊 Database stats: {db_stats}")
        
        # Initialize components (PASS db_client to intent detector)
        self.intent_detector = EnhancedSectorIntentDetector(self.db_client)  # 👈 FIX: Pass db_client
        self.openrouter = OpenRouterIntegration()
        
        # Discover available sectors
        self.available_sectors = self._discover_sectors()
        
        # Cache for sector processors and prompts
        self.sector_processors = {}
        self.sector_prompts = {}
        
        print(f"🤖 Universal Service Bot initialized with {len(self.available_sectors)} sectors")
        print(f"📂 Available sectors: {', '.join(sorted(self.available_sectors))}")

    def _discover_sectors(self) -> List[str]:
        """Discover available sectors from sectors/ directory"""
        sectors = []
        sectors_path = Path(self.sectors_directory)
        
        if not sectors_path.exists():
            print(f"❌ Sectors directory {self.sectors_directory} does not exist")
            return sectors
        
        for item in sectors_path.iterdir():
            if item.is_dir():
                # Check if it has required files
                required_files = ["prompt.txt", "prompt2.txt", "rules.txt"]
                if all((item / f).exists() for f in required_files):
                    sectors.append(item.name)
                    print(f"✅ Discovered sector: {item.name}")
        
        return sorted(sectors)
    
    def get_sector_processor(self, sector: str) -> SectorSpecificProcessor:
        """Get or create a SectorSpecificProcessor for a specific sector using universal database"""
        if sector not in self.sector_processors:
            # Validate sector exists
            if sector not in self.available_sectors:
                raise ValueError(f"Sector '{sector}' not found in available sectors: {', '.join(self.available_sectors)}")
            
            print(f"🔄 Creating processor for sector: {sector}")
            
            try:
                # Create processor using universal database client
                processor = SectorSpecificProcessor(
                    db_client=self.db_client,
                    sector_name=sector
                )
                
                self.sector_processors[sector] = processor
                print(f"✅ Created processor for sector: {sector}")
                
            except Exception as e:
                print(f"❌ Failed to create processor for {sector}: {e}")
                raise
        
        return self.sector_processors[sector]
    
    def load_sector_prompt(self, sector: str) -> str:
        """Load the main prompt for a sector"""
        if sector not in self.sector_prompts:
            sector_path = Path(self.sectors_directory) / sector
            prompt_file = sector_path / "prompt.txt"
            
            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.sector_prompts[sector] = f.read().strip()
                        print(f"✅ Loaded prompt for sector: {sector}")
                except Exception as e:
                    print(f"❌ Error loading prompt for {sector}: {e}")
                    self.sector_prompts[sector] = f"Welcome! How can I help you with {sector.replace('_', ' ')}?"
            else:
                self.sector_prompts[sector] = f"Welcome! How can I help you with {sector.replace('_', ' ')}?"
        
        return self.sector_prompts[sector]
    
    def getIngredients(self, sector: str, user_input: str, orig_prompt: str) -> str:
        """Enhanced getIngredients method following orderChat.py pattern"""
        print(f"[DEBUG] Processing input for sector '{sector}': {user_input}")
        
        try:
            # Get sector-specific processor
            processor = self.get_sector_processor(sector)
            
            # Process the order using sector-specific processor
            processor_output = processor.process_order(user_input)
            
            if not processor_output or processor_output.get("status") == "greeting":
                return orig_prompt
            
            if not processor_output.get("results") and not processor_output.get("item"):
                print("[DEBUG] No relevant information found")
                return orig_prompt
            
            # Build enhanced prompt with menu information
            prompt_update = ""
            results = processor_output.get("results", [])
            category_name = processor_output.get("category", "")
            
            if category_name:
                prompt_update = f"I see you're interested in {category_name}. We have these options:\n\n"
            else:
                prompt_update = "Here are all the matching items:\n\n"
            
            # Group items by category
            categories = {}
            for item in results:
                category = item.get("category", "Other")
                if category not in categories:
                    categories[category] = []
                categories[category].append(item)
            
            # Format items for each category
            for category, items in categories.items():
                if category and len(categories) > 1:
                    prompt_update += f"[Begin Category] {category}\n"
                
                for item in items:
                    item_name = item.get("item", "")
                    base_price = item.get("base_price", item.get("price", 0))
                    
                    item_display = f"- {item_name}: ${base_price:.2f}\n"
                    
                    # Check if this item has rules/customizations
                    if 'selected_rules' in item:
                        try:
                            rules = json.loads(item.get('selected_rules', '[]')) if isinstance(item.get('selected_rules'), str) else item.get('selected_rules', [])
                            if rules:
                                item_display += f"  Requires selections for: {', '.join(rules)}\n"
                        except json.JSONDecodeError:
                            item_display += "  Requires additional selections\n"
                    else:
                        ingredients = item.get("ingredients", "")
                        if ingredients:
                            item_display += f"  {ingredients}\n"
                    
                    prompt_update += item_display
                
                if category and len(categories) > 1:
                    prompt_update += f"[End Category]\n"
            
            prompt_update += "\nWhich option would you like to choose?"
            
            # Handle rule-based items requiring detailed selections
            if processor_output.get("status") == "need_rule_selections":
                if "item" in processor_output:
                    item_name = processor_output.get("item", "")
                    base_price = processor_output.get("base_price", 0)
                    
                    prompt_update = f"Perfect! You're interested in our {item_name} (starts at ${base_price:.2f}).\n\n"
                    prompt_update += "Please let me know your preferences for:\n"
                    
                    for rule_name, options in processor_output.get("available_options", {}).items():
                        if options:
                            prompt_update += f"\n**{rule_name}:**\n"
                            for option in options:
                                for item_option in option.get("items", []):
                                    option_price = item_option.get("price", 0)
                                    price_display = f" (+${option_price:.2f})" if option_price > 0 else ""
                                    prompt_update += f"  • {item_option.get('name', '')}{price_display}\n"
            
            print(f"[DEBUG] Enhanced prompt created for sector '{sector}'")
            return orig_prompt + "\n" + prompt_update
            
        except Exception as e:
            print(f"❌ Error in getIngredients for sector {sector}: {e}")
            return orig_prompt
    
    def chatAway(self, user_input: str, detected_sector: str = None, chat_history: List = None) -> str:
        """Universal chatAway method following orderChat.py pattern"""
        try:
            # Detect sector if not provided
            if not detected_sector:
                detected_sector = self.intent_detector.detect_sector(user_input)
            
            print(f"🎯 Using sector: {detected_sector}")
            
            # Load sector prompt
            sector_prompt = self.load_sector_prompt(detected_sector)
            
            # Enhance prompt with menu information using getIngredients
            enhanced_prompt = self.getIngredients(detected_sector, user_input, sector_prompt)
            
            # Call LLM with enhanced prompt
            if not self.openrouter.llm_local:
                return f"I understand you're interested in {detected_sector.replace('_', ' ')}. How can I help you today? (OpenRouter API not configured)"
            
            # Prepare messages for LLM
            messages = [
                SystemMessage(content=enhanced_prompt),
                *(chat_history or []),
                HumanMessage(content=user_input)
            ]
            
            print(f"🔄 Calling OpenRouter for sector: {detected_sector}")
            response = self.openrouter.llm_local.invoke(messages)
            print(f"✅ OpenRouter response received: {len(response.content)} chars")
            
            return response.content
            
        except Exception as e:
            print(f"❌ Error in chatAway: {e}")
            return f"I'm here to help with {detected_sector.replace('_', ' ') if detected_sector else 'your request'}. Could you tell me more about what you need?"
    
    def process_conversation(self, conversation_messages: List[str]) -> List[str]:
        """Process a full conversation, detecting intent from first message"""
        if not conversation_messages:
            return []
        
        # Detect sector from the first few messages
        detected_sector = self.intent_detector.detect_sector_from_conversation(conversation_messages[:3])
        print(f"🎯 Detected sector from conversation: {detected_sector}")
        
        responses = []
        chat_history = []
        
        for i, message in enumerate(conversation_messages):
            print(f"\n--- Processing Message {i+1}/{len(conversation_messages)} ---")
            
            # Use detected sector for all messages in conversation
            response = self.chatAway(message, detected_sector, chat_history.copy())
            responses.append(response)
            
            # Update chat history
            chat_history.extend([
                HumanMessage(content=message),
                AIMessage(content=response)
            ])
        
        return responses
    
    def get_available_sectors(self) -> List[str]:
        """Get list of available sectors"""
        return self.available_sectors
    
    def get_sector_info(self, sector: str) -> Dict[str, Any]:
        """Get information about a specific sector"""
        if sector not in self.available_sectors:
            return {"error": f"Sector '{sector}' not available"}
        
        return {
            "sector": sector,
            "has_prompt": sector in self.sector_prompts or 
                         (Path(self.sectors_directory) / sector / "prompt.txt").exists(),
            "has_menu": (Path(self.sectors_directory) / sector / "prompt2.txt").exists(),
            "has_rules": (Path(self.sectors_directory) / sector / "rules.txt").exists(),
            "processor_loaded": sector in self.sector_processors
        }
    
    def is_sector_available(self, sector: str) -> bool:
        """Check if a sector is available"""
        return sector in self.available_sectors

##########################
# MAIN FUNCTION
##########################

def main():
    """Example usage of the Universal Service Bot"""
    print("🤖 Universal Service Bot with Universal Database Integration")
    print("=" * 70)
    
    # Initialize bot
    try:
        bot = UniversalServiceBot()
    except RuntimeError as e:
        print(f"❌ {e}")
        print("💡 Please ensure the universal database is indexed first:")
        print("   python universal_service_bot.py --index-all")
        return
    
    if not bot.get_available_sectors():
        print("❌ No sectors found. Please create sectors/ directory with sector folders.")
        return
    
    # Interactive chat
    print(f"\n🗣️ Interactive Chat (Available sectors: {', '.join(bot.get_available_sectors())})")
    print("Type 'quit' to exit\n")
    
    chat_history = []
    detected_sector = None
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Detect sector from first message or use previous
            if not detected_sector:
                detected_sector = bot.intent_detector.detect_sector(user_input)
                print(f"🎯 Detected sector: {detected_sector}")
            
            # Get response
            response = bot.chatAway(user_input, detected_sector, chat_history)
            print(f"Bot: {response}")
            
            # Update chat history
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response)
            ])
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
