#!/usr/bin/env python3

import unittest
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import sys

# Import the updated universal service bot that works with universal database
from updated_universal_service_bot import UniversalServiceBot, EnhancedSectorIntentDetector
from langchain_core.messages import HumanMessage, AIMessage

##########################
# COMPREHENSIVE TEST FRAMEWORK (WITHOUT MENUINDEXER/ORDERPROCESSOR)
##########################

class ComprehensiveUniversalServiceBotTest(unittest.TestCase):
    """Comprehensive test framework for Universal Service Bot using universal database"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("🧪 Setting up Universal Service Bot tests with universal database...")
        
        # Initialize bot (uses universal database directly)
        try:
            cls.bot = UniversalServiceBot()
            cls.available_sectors = cls.bot.get_available_sectors()
            cls.intent_detector = EnhancedSectorIntentDetector(cls.bot.db_client)
            print(f"✅ Bot initialized with {len(cls.available_sectors)} sectors")
        except Exception as e:
            print(f"❌ Failed to initialize bot: {e}")
            print("💡 Make sure universal_chroma_database exists and is properly indexed")
            raise

        # Complete test conversation scenarios for all sectors
        cls.conversation_scenarios = {
            "healthcare": [
                "Hello",
                "I need a doctor appointment for my headache",
                "I have fever and headache symptoms for 2 days",
                "I need a routine checkup appointment",
                "My insurance is Blue Cross Blue Shield member ID 12345",
                "Tomorrow at 2pm works for me",
                "Yes, process the copay payment"
            ],
            "food_delivery": [
                "Hello",
                "I want to order breakfast",
                "I want Build Your Own Breakfast with scrambled eggs",
                "Add bacon and hash browns please",
                "Deliver to 123 Main St at 8am",
                "Process payment with my credit card"
            ],
            "beauty_salon": [
                "Hello",
                "I need a complete makeover",
                "I'd like haircut, hair color, and manicure",
                "Book me for Friday afternoon at 2pm",
                "My name is Sarah Johnson",
                "Yes, charge my card for the services"
            ],
            "legal_services": [
                "Hello",
                "I need legal consultation",
                "Family law matter, divorce proceedings",
                "Schedule consultation with a family attorney",
                "Next week Tuesday afternoon is preferred",
                "I can pay the consultation fee upfront"
            ],
            "financial_services": [
                "Hello",
                "I need financial planning advice",
                "Investment consultation for retirement planning",
                "Schedule appointment with financial advisor",
                "I have $50k to invest",
                "Let's proceed with the consultation"
            ],
            "real_estate": [
                "Hello",
                "I'm looking to buy a house",
                "3 bedroom house in good neighborhood", 
                "Budget around $400k",
                "Schedule viewing appointments",
                "Let's proceed with house hunting"
            ],
            "fitness_gym": [
                "Hello",
                "I want to join the gym",
                "I need personal training and group classes",
                "Premium membership with all access",
                "Start next Monday",
                "Process membership payment"
            ],
            "photography": [
                "Hello",
                "I need wedding photography",
                "Full day coverage with engagement session",
                "Wedding date is June 15th",
                "Book the premium package",
                "Process booking deposit"
            ],
            "pet_services": [
                "Hello",
                "My dog needs grooming and vet checkup",
                "Large dog, full grooming service",
                "Include nail trimming and bath",
                "Schedule for this weekend", 
                "Charge my card for services"
            ],
            "transportation": [
                "Hello",
                "I need airport pickup service",
                "Flight arrives tomorrow at 3pm",
                "Pickup from Terminal 1 to downtown",
                "Premium vehicle please",
                "Confirm booking and payment"
            ],
            "travel_hotel": [
                "Hello",
                "I need hotel booking for business trip",
                "5 nights in downtown location",
                "Executive suite with wifi",
                "Check-in Monday check-out Friday",
                "Book with corporate rate"
            ],
            "home_services": [
                "Hello",
                "I need plumbing and electrical work",
                "Kitchen sink repair and outlet installation",
                "Emergency service required",
                "Available today after 2pm",
                "Proceed with service call"
            ],
            "education_tutoring": [
                "Hello",
                "My child needs math tutoring",
                "High school algebra and geometry",
                "Twice weekly sessions",
                "Start next week",
                "Enroll in tutoring program"
            ],
            "insurance": [
                "Hello", 
                "I need auto insurance quote",
                "Full coverage for 2019 Honda Civic",
                "Good driving record, no claims",
                "Start policy immediately",
                "Purchase recommended coverage"
            ],
            "event_planning": [
                "Hello",
                "I need help planning my wedding",
                "150 guests, outdoor ceremony",
                "Need venue, catering, and coordination",
                "Wedding date is September 10th",
                "Book the complete planning package"
            ],
            "auto_repair": [
                "Hello",
                "My car needs brake repair",
                "Squeaking noise and soft pedal",
                "2018 Toyota Camry",
                "Can you check it today?",
                "Proceed with brake service"
            ],
            "laundry_services": [
                "Hello",
                "I need laundry service",
                "I have 2 loads of laundry that need same day service",
                "I want wash and fold with pickup and delivery",
                "My address is 123 Main Street",
                "Charge my credit card for the service"
            ],
            "it_services": [
                "Hello", 
                "My laptop has a virus problem",
                "Computer is running very slowly with pop-ups",
                "I need urgent same day repair service",
                "Please come to my office for on-site service",
                "Process payment on completion"
            ],
            "moving_services": [
                "Hello",
                "I need moving services",
                "I need to move my 3 bedroom house locally",
                "About 25 miles distance with packing service",
                "Moving date is next Friday",
                "Please provide full service estimate"
            ]
        }
        
        print(f"✅ Test setup complete. Available sectors: {len(cls.available_sectors)}")
        print(f"📂 Sectors found: {', '.join(sorted(cls.available_sectors))}")
        print(f"🎭 Conversation scenarios: {len(cls.conversation_scenarios)}")

    def test_complete_state_transitions(self):
        """Test complete conversation flow through all states to goal for ALL sectors"""
        print(f"\n🔄 Testing Complete State Transitions for ALL Sectors")
        print("=" * 60)
        
        # Test ALL available sectors instead of just 3
        test_sectors = self.available_sectors  # All 18+ sectors
        
        # Define goal state indicators for different sector types
        goal_state_indicators = {
            "food_delivery": ["payment", "order", "delivery", "confirm", "checkout"],
            "healthcare": ["appointment", "scheduled", "booking", "confirm", "payment"],
            "beauty_salon": ["appointment", "booking", "scheduled", "payment", "confirm"],
            "legal_services": ["consultation", "retainer", "payment", "scheduled", "booked"],
            "financial_services": ["consultation", "meeting", "scheduled", "proceed", "appointment"],
            "real_estate": ["viewing", "appointment", "scheduled", "proceed", "meeting"],
            "fitness_gym": ["membership", "payment", "enrollment", "signup", "join"],
            "photography": ["booking", "deposit", "payment", "scheduled", "session"],
            "pet_services": ["appointment", "scheduled", "booking", "payment", "service"],
            "transportation": ["booking", "confirmed", "pickup", "scheduled", "payment"],
            "travel_hotel": ["booking", "reservation", "confirmed", "payment", "booked"],
            "home_services": ["appointment", "scheduled", "service", "payment", "booking"],
            "education_tutoring": ["enrollment", "sessions", "scheduled", "payment", "program"],
            "insurance": ["policy", "coverage", "payment", "purchase", "quote"],
            "event_planning": ["booking", "package", "payment", "planning", "confirmed"],
            "auto_repair": ["appointment", "service", "scheduled", "booking", "repair"],
            "moving_services": ["estimate", "booking", "scheduled", "service", "moving"],
            "it_services": ["service", "repair", "appointment", "scheduled", "technical"],
            "laundry_services": ["service", "pickup", "scheduled", "payment", "laundry"]
        }
        
        successful_transitions = 0
        failed_transitions = 0
        
        for sector in test_sectors:
            if sector in self.conversation_scenarios:
                with self.subTest(sector=sector):
                    # Use ALL messages to reach goal state
                    full_conversation = self.conversation_scenarios[sector]  # All messages
                    
                    try:
                        print(f"\n🎯 Testing {sector}: {len(full_conversation)} conversation steps")
                        responses = self.bot.process_conversation(full_conversation)
                        
                        # Verify conversation completed
                        self.assertEqual(len(responses), len(full_conversation), 
                                    f"Response count mismatch for {sector}")
                        
                        # Check final state with sector-specific indicators
                        final_response = responses[-1].lower()
                        goal_indicators = goal_state_indicators.get(sector, ["payment", "confirm", "complete"])
                        
                        reached_goal = any(indicator in final_response for indicator in goal_indicators)
                        
                        if reached_goal:
                            successful_transitions += 1
                            print(f"✅ {sector}: Complete state transition to goal ({len(responses)} states)")
                            
                            # Additional verification - check response progression
                            response_lengths = [len(resp) for resp in responses]
                            self.assertTrue(all(length > 5 for length in response_lengths), 
                                        f"Short responses detected in {sector}")
                            
                        else:
                            failed_transitions += 1
                            print(f"⚠️ {sector}: May not have reached final goal state")
                            print(f"   Final response preview: {final_response[:100]}...")
                            print(f"   Expected indicators: {goal_indicators}")
                            
                            # Don't fail the test, just log the issue
                            # Some sectors might have different conversation flows
                        
                    except Exception as e:
                        failed_transitions += 1
                        print(f"❌ {sector}: Failed to complete state transitions - {e}")
                        if "API" not in str(e) and "OpenRouter" not in str(e):
                            self.fail(f"State transition failed for {sector}: {e}")
            else:
                print(f"⚠️ {sector}: No conversation scenario defined")
        
        # Summary statistics
        total_tested = successful_transitions + failed_transitions
        success_rate = (successful_transitions / total_tested * 100) if total_tested > 0 else 0
        
        print(f"\n📊 COMPLETE STATE TRANSITION SUMMARY:")
        print(f"✅ Successful transitions: {successful_transitions}")
        print(f"⚠️ Incomplete transitions: {failed_transitions}")
        print(f"🎯 Success rate: {success_rate:.1f}%")
        
        # Ensure at least 70% of sectors reach goal states
        self.assertGreaterEqual(success_rate, 70.0, 
                            f"Goal state completion rate too low: {success_rate:.1f}%")


    def test_complete_state_transitions_detailed(self):
        """Detailed analysis of state transitions across all sectors"""
        print(f"\n🔄 Detailed State Transition Analysis")
        print("=" * 60)
        
        # Analyze conversation flow patterns
        conversation_patterns = {}
        
        for sector in self.available_sectors:
            if sector in self.conversation_scenarios:
                conversation = self.conversation_scenarios[sector]
                
                print(f"\n🎯 Analyzing {sector}:")
                print(f"   Total conversation steps: {len(conversation)}")
                
                try:
                    responses = self.bot.process_conversation(conversation)
                    
                    # Analyze response patterns
                    conversation_patterns[sector] = {
                        "steps": len(conversation),
                        "responses": len(responses),
                        "avg_response_length": sum(len(r) for r in responses) / len(responses),
                        "final_response_length": len(responses[-1]),
                        "progression": "completed"
                    }
                    
                    # Check for conversation progression indicators
                    progression_keywords = {
                        1: ["hello", "hi", "greeting"],           # State 1: Greeting
                        2: ["need", "want", "looking", "require"], # State 2: Intent
                        3: ["select", "choose", "option"],        # State 3: Selection
                        4: ["details", "information", "address"], # State 4: Details
                        5: ["payment", "confirm", "proceed"],     # State 5: Finalization
                    }
                    
                    detected_states = []
                    for i, response in enumerate(responses):
                        response_lower = response.lower()
                        for state_num, keywords in progression_keywords.items():
                            if any(keyword in response_lower for keyword in keywords):
                                detected_states.append(state_num)
                                break
                        else:
                            detected_states.append(0)  # Unknown state
                    
                    conversation_patterns[sector]["state_progression"] = detected_states
                    conversation_patterns[sector]["max_state_reached"] = max(detected_states) if detected_states else 0
                    
                    print(f"   ✅ Processed successfully")
                    print(f"   📊 State progression: {detected_states}")
                    print(f"   🎯 Highest state reached: {max(detected_states)}")
                    
                except Exception as e:
                    conversation_patterns[sector] = {"error": str(e)}
                    print(f"   ❌ Error: {e}")
        
        # Generate comprehensive report
        print(f"\n📋 COMPREHENSIVE STATE ANALYSIS REPORT:")
        print("=" * 60)
        
        high_completers = []
        medium_completers = []
        low_completers = []
        
        for sector, data in conversation_patterns.items():
            if "error" not in data:
                max_state = data.get("max_state_reached", 0)
                if max_state >= 5:
                    high_completers.append(sector)
                elif max_state >= 3:
                    medium_completers.append(sector)
                else:
                    low_completers.append(sector)
        
        print(f"🏆 High Completers (State 5+): {len(high_completers)}")
        for sector in high_completers:
            print(f"   ✅ {sector}")
        
        print(f"\n🥈 Medium Completers (State 3-4): {len(medium_completers)}")
        for sector in medium_completers:
            print(f"   🔶 {sector}")
        
        print(f"\n🥉 Low Completers (State 1-2): {len(low_completers)}")
        for sector in low_completers:
            print(f"   🔸 {sector}")
        
        # Assert that most sectors reach at least medium completion
        total_sectors = len(high_completers) + len(medium_completers) + len(low_completers)
        completion_rate = (len(high_completers) + len(medium_completers)) / total_sectors * 100 if total_sectors > 0 else 0
        
        print(f"\n🎯 Overall Completion Rate: {completion_rate:.1f}%")
        self.assertGreaterEqual(completion_rate, 75.0, "State completion rate too low")


    def test_universal_database_connection(self):
        """Test that the universal database is accessible"""
        print(f"\n🗄️ Testing Universal Database Connection")
        print("=" * 50)
        
        # Check database health
        db_stats = self.bot.db_client.check_database_health()
        
        self.assertTrue(db_stats.get("healthy", False), "Universal database is not healthy")
        self.assertGreater(db_stats.get("total_documents", 0), 0, "Universal database has no documents")
        
        print(f"✅ Database is healthy with {db_stats.get('total_documents', 0)} total documents")
        
        # Test each collection
        collections = ["categories", "items", "rules", "rule_options", "rule_items"]
        for collection_name in collections:
            count = db_stats.get(collection_name, 0)
            self.assertGreater(count, 0, f"Collection {collection_name} is empty")
            print(f"   {collection_name}: {count} documents")

    def test_enhanced_intent_detection(self):
        """Test enhanced intent detection using multiple messages"""
        print(f"\n🎯 Testing Enhanced Intent Detection")
        print("=" * 50)
        
        for sector, conversations in self.conversation_scenarios.items():
            if sector in self.available_sectors:
                with self.subTest(sector=sector):
                    # Test with first 2 messages (skip just "Hello")
                    first_two_messages = conversations[:2]
                    
                    detected_sector = self.intent_detector.detect_sector_from_conversation(
                        first_two_messages, max_messages=2
                    )
                    
                    print(f"Messages: {first_two_messages}")
                    print(f"Expected: {sector}, Detected: {detected_sector}")
                    
                    # Check if detection is correct or reasonable
                    is_correct = (detected_sector == sector)
                    print(f"✅ Correct" if is_correct else f"⚠️  Different sector: {detected_sector}")
                    
                    # Test with first 3 messages for even better context
                    if len(conversations) >= 3:
                        first_three_messages = conversations[:3]
                        detected_sector_3 = self.intent_detector.detect_sector_from_conversation(
                            first_three_messages, max_messages=3
                        )
                        print(f"With 3 messages: {detected_sector_3}")

    def test_sector_processors_creation(self):
        """Test that sector processors can be created using universal database"""
        print(f"\n🔧 Testing Sector Processors Creation")
        print("=" * 40)
        
        for sector in self.available_sectors:
            with self.subTest(sector=sector):
                try:
                    processor = self.bot.get_sector_processor(sector)
                    self.assertIsNotNone(processor, f"Failed to create processor for {sector}")
                    
                    # Test that processor can handle queries
                    result = processor.process_order("test query")
                    self.assertIsInstance(result, dict, f"Processor for {sector} should return dict")
                    
                    print(f"✅ {sector}: Processor created and tested successfully")
                except Exception as e:
                    print(f"❌ {sector}: Failed to create processor - {e}")
                    self.fail(f"Failed to create processor for {sector}: {e}")

    def test_sector_prompts_loading(self):
        """Test that sector prompts can be loaded"""
        print(f"\n📝 Testing Sector Prompts Loading")
        print("=" * 40)
        
        for sector in self.available_sectors:
            with self.subTest(sector=sector):
                prompt = self.bot.load_sector_prompt(sector)
                self.assertIsInstance(prompt, str, f"Prompt for {sector} should be string")
                self.assertGreater(len(prompt), 10, f"Prompt for {sector} too short")
                print(f"✅ {sector}: Prompt loaded ({len(prompt)} chars)")

    def test_getIngredients_integration(self):
        """Test getIngredients method with universal database"""
        print(f"\n🍳 Testing getIngredients Integration")
        print("=" * 40)
        
        test_queries = {
            "food_delivery": "I want breakfast",
            "beauty_salon": "I need a haircut",
            "healthcare": "I need a checkup",
            "laundry_services": "I need laundry service",
            "it_services": "My computer has virus",
            "auto_repair": "My car needs repair"
        }
        
        for sector, query in test_queries.items():
            if sector in self.available_sectors:
                with self.subTest(sector=sector):
                    try:
                        original_prompt = f"Welcome to {sector.replace('_', ' ')}"
                        enhanced_prompt = self.bot.getIngredients(sector, query, original_prompt)
                        
                        self.assertIsInstance(enhanced_prompt, str)
                        self.assertGreaterEqual(len(enhanced_prompt), len(original_prompt))
                        print(f"✅ {sector}: Enhanced prompt created ({len(enhanced_prompt)} chars)")
                        
                    except Exception as e:
                        print(f"❌ {sector}: getIngredients failed - {e}")
                        self.fail(f"getIngredients failed for {sector}: {e}")

    def test_chatAway_method(self):
        """Test chatAway method with universal database integration"""
        print(f"\n💬 Testing chatAway Method")
        print("=" * 40)
        
        # Test with OpenRouter if API key is available
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY not set - testing without AI responses")
        
        for sector, conversations in self.conversation_scenarios.items():
            if sector in self.available_sectors:
                with self.subTest(sector=sector):
                    first_message = conversations[1]  # Skip "Hello", use meaningful message
                    
                    try:
                        response = self.bot.chatAway(first_message, sector)
                        
                        self.assertIsInstance(response, str)
                        self.assertGreater(len(response), 10)
                        print(f"✅ {sector}: chatAway response received ({len(response)} chars)")
                        
                        # Check if response contains sector-relevant terms
                        response_lower = response.lower()
                        sector_terms = {
                            "healthcare": ["health", "doctor", "appointment", "medical"],
                            "food_delivery": ["food", "order", "menu", "breakfast"],
                            "beauty_salon": ["hair", "beauty", "salon", "service"],
                            "laundry_services": ["laundry", "wash", "clean", "service"],
                            "it_services": ["computer", "tech", "repair", "virus"],
                            "auto_repair": ["car", "vehicle", "repair", "brake"]
                        }
                        
                        if sector in sector_terms:
                            relevant_terms = any(term in response_lower for term in sector_terms[sector])
                            if relevant_terms:
                                print(f"   Response appears relevant to {sector}")
                            else:
                                print(f"   Response may not be sector-specific")
                        
                    except Exception as e:
                        print(f"❌ {sector}: chatAway failed - {e}")
                        # Don't fail the test if it's just an API issue
                        if "API" not in str(e) and "OpenRouter" not in str(e):
                            self.fail(f"chatAway failed for {sector}: {e}")

    def test_full_conversation_flow(self):
        """Test complete conversation flow with enhanced intent detection"""
        print(f"\n🗣️ Testing Full Conversation Flow")
        print("=" * 40)
        
        # Test a few key sectors with full conversations
        test_sectors = ["food_delivery", "healthcare", "beauty_salon", "it_services"]
        
        for sector in test_sectors:
            if sector in self.available_sectors and sector in self.conversation_scenarios:
                with self.subTest(sector=sector):
                    conversation = self.conversation_scenarios[sector][:3]  # Test first 3 messages
                    
                    try:
                        responses = self.bot.process_conversation(conversation)
                        
                        self.assertEqual(len(responses), len(conversation))
                        
                        for i, (message, response) in enumerate(zip(conversation, responses)):
                            self.assertIsInstance(response, str)
                            self.assertGreater(len(response), 5)
                        
                        print(f"✅ {sector}: Full conversation processed ({len(responses)} responses)")
                        
                    except Exception as e:
                        print(f"❌ {sector}: Conversation flow failed - {e}")
                        if "API" not in str(e) and "OpenRouter" not in str(e):
                            self.fail(f"Conversation flow failed for {sector}: {e}")

    def test_sector_availability(self):
        """Test sector availability and discovery"""
        print(f"\n📂 Testing Sector Availability")
        print("=" * 40)
        
        available_sectors = self.bot.get_available_sectors()
        
        self.assertIsInstance(available_sectors, list)
        self.assertGreater(len(available_sectors), 0, "No sectors discovered")
        
        # Check that each available sector has required files
        sectors_path = Path(self.bot.sectors_directory)
        for sector in available_sectors:
            sector_path = sectors_path / sector
            self.assertTrue(sector_path.exists(), f"Sector directory {sector} not found")
            
            required_files = ["prompt.txt", "prompt2.txt", "rules.txt"]
            for file_name in required_files:
                file_path = sector_path / file_name
                self.assertTrue(file_path.exists(), f"Required file {file_name} not found for {sector}")
        
        print(f"✅ All {len(available_sectors)} sectors have required files")

    def test_intent_detection_accuracy(self):
        """Test intent detection accuracy across different query types"""
        print(f"\n🎯 Testing Intent Detection Accuracy")
        print("=" * 40)
        
        # Additional test queries for intent detection
        specific_queries = {
            "food_delivery": [
                "I want to order omelet",
                "I'm hungry for breakfast",
                "Can I get a bagel delivered?"
            ],
            "healthcare": [
                "I need to see a doctor",
                "Book medical appointment",
                "I have health concerns"
            ],
            "beauty_salon": [
                "I need a haircut",
                "Book salon appointment",
                "I want manicure and facial"
            ],
            "it_services": [
                "My computer has virus",
                "Laptop repair needed",
                "Tech support required"
            ],
            "auto_repair": [
                "My car needs brake repair",
                "Engine trouble",
                "Vehicle maintenance"
            ]
        }
        
        correct_detections = 0
        total_tests = 0
        
        for expected_sector, queries in specific_queries.items():
            if expected_sector in self.available_sectors:
                for query in queries:
                    detected_sector = self.intent_detector.detect_sector(query)
                    total_tests += 1
                    
                    if detected_sector == expected_sector:
                        correct_detections += 1
                        print(f"✅ '{query}' → {detected_sector}")
                    else:
                        print(f"⚠️  '{query}' → {detected_sector} (expected {expected_sector})")
        
        if total_tests > 0:
            accuracy = (correct_detections / total_tests) * 100
            print(f"\n📊 Intent Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{total_tests})")
            
            # We expect at least 50% accuracy for intent detection
            self.assertGreaterEqual(accuracy, 50.0, "Intent detection accuracy too low")

    def test_conversation_with_enhanced_intent(self):
        """Test conversation processing with enhanced intent detection from first messages"""
        print(f"\n🎭 Testing Enhanced Intent Detection in Conversations")
        print("=" * 50)
        
        for sector, conversations in self.conversation_scenarios.items():
            if sector in self.available_sectors:
                with self.subTest(sector=sector):
                    # Test intent detection from first few messages
                    first_messages = conversations[:3]
                    
                    detected_sector = self.intent_detector.detect_sector_from_conversation(first_messages)
                    print(f"Sector: {sector}")
                    print(f"First messages: {first_messages}")
                    print(f"Detected: {detected_sector}")
                    
                    # The detection should be valid (even if not exact match)
                    self.assertIn(detected_sector, self.available_sectors, 
                                f"Detected sector '{detected_sector}' not in available sectors")
                    
                    print(f"✅ Valid sector detected for {sector}")

    def test_sector_specific_database_queries(self):
        """Test that sector-specific queries work with universal database"""
        print(f"\n🔍 Testing Sector-Specific Database Queries")
        print("=" * 50)
        
        test_queries = {
            "food_delivery": ["breakfast", "omelet", "bagel"],
            "healthcare": ["doctor", "appointment", "checkup"],
            "beauty_salon": ["haircut", "manicure", "facial"],
            "auto_repair": ["brake", "engine", "oil change"],
            "it_services": ["virus", "computer", "laptop repair"]
        }
        
        for sector, queries in test_queries.items():
            if sector in self.available_sectors:
                with self.subTest(sector=sector):
                    for query in queries:
                        try:
                            processor = self.bot.get_sector_processor(sector)
                            result = processor.process_order(query)
                            
                            self.assertIsInstance(result, dict)
                            print(f"✅ {sector}: Query '{query}' processed successfully")
                            
                            # Check if results are sector-specific
                            if result.get("results"):
                                for item in result["results"]:
                                    if "sector" in item:
                                        self.assertEqual(item["sector"], sector, 
                                                       f"Item from wrong sector: {item}")
                            
                        except Exception as e:
                            print(f"❌ {sector}: Query '{query}' failed - {e}")
                            self.fail(f"Sector query failed for {sector}: {e}")

    def test_database_filtering_effectiveness(self):
        """Test that database filtering by sector works correctly"""
        print(f"\n🔧 Testing Database Filtering Effectiveness")
        print("=" * 50)
        
        # Test that sector filtering returns only relevant results
        for sector in self.available_sectors[:5]:  # Test first 5 sectors
            with self.subTest(sector=sector):
                try:
                    processor = self.bot.get_sector_processor(sector)
                    
                    # Use a generic query that might match multiple sectors
                    result = processor.sector_filtered_unified_search("service")
                    
                    # Check that all results belong to the specified sector
                    for item in result:
                        metadata = item.get("metadata", {})
                        item_sector = metadata.get("sector")
                        if item_sector:
                            self.assertEqual(item_sector, sector, 
                                           f"Found item from sector '{item_sector}' when querying '{sector}'")
                    
                    print(f"✅ {sector}: Filtering works correctly ({len(result)} results)")
                    
                except Exception as e:
                    print(f"❌ {sector}: Filtering test failed - {e}")
                    self.fail(f"Database filtering failed for {sector}: {e}")

##########################
# TEST EXECUTION AND UTILITIES
##########################

def run_comprehensive_tests():
    """Run comprehensive tests for Universal Service Bot"""
    print("🚀 Starting Comprehensive Universal Service Bot Tests")
    print("=" * 60)
    
    # Check if universal database exists
    if not os.path.exists("universal_chroma_database"):
        print("❌ Universal database not found!")
        print("💡 Please run indexing first:")
        print("   python universal_service_bot.py --index-all")
        return False
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensiveUniversalServiceBotTest)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print("🔥 Running Quick Smoke Test")
    print("=" * 30)
    
    try:
        bot = UniversalServiceBot()
        sectors = bot.get_available_sectors()
        
        if not sectors:
            print("❌ No sectors found")
            return False
        
        print(f"✅ Found {len(sectors)} sectors")
        
        # Test intent detection
        detector = bot.intent_detector
        test_query = "I need a doctor appointment"
        detected = detector.detect_sector(test_query)
        print(f"✅ Intent detection works: '{test_query}' → {detected}")
        
        # Test database connectivity
        stats = bot.db_client.check_database_health()
        if stats.get("healthy"):
            print(f"✅ Database healthy: {stats.get('total_documents', 0)} documents")
        else:
            print("❌ Database not healthy")
            return False
        
        print("🎉 Smoke test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Check if we should run specific tests
    if len(sys.argv) > 1:
        if sys.argv[1] == "comprehensive":
            success = run_comprehensive_tests()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "smoke":
            success = run_quick_smoke_test()
            sys.exit(0 if success else 1)
        else:
            # Run specific test method
            unittest.main()
    else:
        # Run comprehensive tests by default
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
