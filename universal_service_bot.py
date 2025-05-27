#!/usr/bin/env python3

import os
import json
import re
from pathlib import Path
from typing import Annotated, Dict, Any, List, Optional, TypedDict
import operator
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

##########################
# UNIVERSAL STATE DEFINITIONS
##########################

class UniversalState(TypedDict):
    messages: Annotated[list, add_messages]
    sector_context: str
    current_step: str
    user_info: Dict[str, Any]
    selected_items: Annotated[list, operator.add]
    session_data: Dict[str, Any]
    qualified: bool
    payment_processed: bool
    completed: bool

class UniversalWorkflowStates:
    """Universal workflow states that work across ALL sectors"""
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"
    SELECTION = "selection"
    CUSTOMIZATION = "customization"
    INFO_COLLECTION = "info_collection"
    VERIFICATION = "verification"
    PAYMENT = "payment"
    FINALIZATION = "finalization"
    MODIFICATION = "modification"
    CANCELLATION = "cancellation"

##########################
# FILE LOADER
##########################

class FileLoader:
    """Load sector prompts and rules from files"""
    
    @staticmethod
    def load_sector_prompts_from_files(prompts_directory: str) -> Dict[str, str]:
        """Load sector prompts from text files"""
        sector_prompts = {}
        prompts_path = Path(prompts_directory)
        
        if not prompts_path.exists():
            print(f"❌ Prompts directory {prompts_directory} does not exist")
            return sector_prompts
        
        # Load .txt files as sector prompts
        for file_path in prompts_path.glob("*.txt"):
            sector_name = file_path.stem
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sector_prompt = f.read().strip()
                    sector_prompts[sector_name] = sector_prompt
                    print(f"✅ Loaded prompt for '{sector_name}' from {file_path.name}")
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")
        
        return sector_prompts
    
    @staticmethod
    def load_sector_rules_from_files(rules_directory: str) -> Dict[str, Dict[str, Any]]:
        """Load sector selection rules from JSON files"""
        sector_rules = {}
        rules_path = Path(rules_directory)
        
        if not rules_path.exists():
            print(f"❌ Rules directory {rules_directory} does not exist")
            return sector_rules
        
        for file_path in rules_path.glob("*.json"):
            sector_name = file_path.stem
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    sector_rules[sector_name] = rules_data
                    print(f"✅ Loaded rules for '{sector_name}' from {file_path.name}")
            except Exception as e:
                print(f"❌ Error loading rules {file_path.name}: {e}")
        
        return sector_rules

##########################
# PROMPT PARSER
##########################

class SectorPromptParser:
    """Parse sector prompts into workflow state prompts"""
    
    def __init__(self):
        self.state_indicators = {
            UniversalWorkflowStates.DISCOVERY: [
                'welcome', 'available', 'options', 'choose from', 'what type', 'services', 'menu',
                'browse', 'explore', 'find', 'search', 'looking for', 'interested in'
            ],
            UniversalWorkflowStates.QUALIFICATION: [
                'describe', 'symptoms', 'problem', 'issue', 'tell me about', 'what brings you',
                'requirements', 'level', 'experience', 'condition', 'need help with'
            ],
            UniversalWorkflowStates.SELECTION: [
                'here are', 'choose', 'select', 'pick', 'available slots', 'appointments',
                'doctors', 'specialists', 'services offered', 'packages', '$', 'price'
            ],
            UniversalWorkflowStates.CUSTOMIZATION: [
                'special requests', 'preferences', 'customize', 'modify', 'instructions',
                'allergies', 'dietary', 'style', 'color', 'specific'
            ],
            UniversalWorkflowStates.INFO_COLLECTION: [
                'address', 'delivery', 'contact', 'phone', 'email', 'schedule', 'appointment',
                'when would you', 'where should', 'time', 'date', 'location'
            ],
            UniversalWorkflowStates.VERIFICATION: [
                'insurance', 'verify', 'confirm', 'check', 'validate', 'estimate',
                'quote', 'proceed', 'approve'
            ],
            UniversalWorkflowStates.PAYMENT: [
                'payment', 'pay', 'checkout', 'billing', 'credit card', 'charge',
                'process payment', 'copay', 'cost', 'total'
            ],
            UniversalWorkflowStates.FINALIZATION: [
                'confirmed', 'complete', 'finished', 'booked', 'scheduled', 'thank you',
                'success', 'done', 'see you', 'arrival time'
            ]
        }
    
    def parse_sector_prompt(self, sector_prompt: str) -> Dict[str, str]:
        """Parse a sector prompt into workflow state prompts"""
        sentences = self._split_into_sentences(sector_prompt)
        sentence_assignments = {}
        
        for sentence in sentences:
            best_state = self._find_best_state_for_sentence(sentence)
            if best_state not in sentence_assignments:
                sentence_assignments[best_state] = []
            sentence_assignments[best_state].append(sentence)
        
        state_prompts = {}
        for state in [UniversalWorkflowStates.DISCOVERY, UniversalWorkflowStates.QUALIFICATION,
                     UniversalWorkflowStates.SELECTION, UniversalWorkflowStates.CUSTOMIZATION,
                     UniversalWorkflowStates.INFO_COLLECTION, UniversalWorkflowStates.VERIFICATION,
                     UniversalWorkflowStates.PAYMENT, UniversalWorkflowStates.FINALIZATION]:
            
            if state in sentence_assignments:
                state_prompts[state] = ' '.join(sentence_assignments[state])
            else:
                state_prompts[state] = self._generate_default_prompt(state)
        
        state_prompts[UniversalWorkflowStates.MODIFICATION] = "What would you like to modify or change?"
        state_prompts[UniversalWorkflowStates.CANCELLATION] = "Your request has been cancelled. Thank you!"
        
        return state_prompts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]\s*|\n', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _find_best_state_for_sentence(self, sentence: str) -> str:
        sentence_lower = sentence.lower()
        state_scores = {}
        
        for state, keywords in self.state_indicators.items():
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            state_scores[state] = score
        
        if state_scores and max(state_scores.values()) > 0:
            return max(state_scores, key=state_scores.get)
        
        # Fallback logic
        if any(word in sentence_lower for word in ['$', 'price', 'cost']):
            return UniversalWorkflowStates.SELECTION
        elif any(word in sentence_lower for word in ['welcome', 'available', 'menu']):
            return UniversalWorkflowStates.DISCOVERY
        else:
            return UniversalWorkflowStates.DISCOVERY
    
    def _generate_default_prompt(self, state: str) -> str:
        defaults = {
            UniversalWorkflowStates.DISCOVERY: "What can I help you with today?",
            UniversalWorkflowStates.QUALIFICATION: "Please tell me more about your requirements.",
            UniversalWorkflowStates.SELECTION: "Here are the available options for you to choose from.",
            UniversalWorkflowStates.CUSTOMIZATION: "Any special preferences or customizations?",
            UniversalWorkflowStates.INFO_COLLECTION: "Please provide the necessary details.",
            UniversalWorkflowStates.VERIFICATION: "Let me verify the information.",
            UniversalWorkflowStates.PAYMENT: "Let's process the payment.",
            UniversalWorkflowStates.FINALIZATION: "Everything is complete! Thank you!"
        }
        return defaults.get(state, "Please continue.")

##########################
# GENERIC SELECTION RULES
##########################

class GenericSelectionRules:
    """Universal rule-based selection system for ANY industry sector"""
    
    def __init__(self):
        self.sector_rules = {}
    
    def load_rules_from_config(self, sector: str, rules_config: Dict[str, Any]):
        """Load rules for any sector from configuration"""
        self.sector_rules[sector] = rules_config
    
    def get_customizable_items(self, sector: str) -> List[str]:
        """Get items that have custom selection rules"""
        return list(self.sector_rules.get(sector, {}).keys())
    
    def get_selection_categories(self, sector: str, item: str) -> Dict[str, Any]:
        """Get selection categories for any item in any sector"""
        if sector not in self.sector_rules or item not in self.sector_rules[sector]:
            return {}
        
        item_config = self.sector_rules[sector][item]
        categories = {}
        
        for category_name, category_config in item_config.items():
            if category_name == "base_price":
                continue
                
            categories[category_name] = {
                "display_name": category_config.get("display_name", category_name.title()),
                "options": category_config.get("options", []),
                "min_selections": category_config.get("min_selections", 0),
                "max_selections": category_config.get("max_selections", 999),
                "required": category_config.get("required", False),
                "exclusive": category_config.get("exclusive", False),
                "pricing": category_config.get("pricing", {}),
                "description": category_config.get("description", "")
            }
        
        return categories
    
    def validate_selections(self, sector: str, item: str, selections: Dict[str, List[str]]) -> Dict[str, Any]:
        """Universal validation for any sector's selections"""
        categories = self.get_selection_categories(sector, item)
        errors = []
        
        for category_name, category_config in categories.items():
            selected = selections.get(category_name, [])
            
            if len(selected) < category_config["min_selections"]:
                errors.append(f"{category_config['display_name']}: Minimum {category_config['min_selections']} required")
            
            if len(selected) > category_config["max_selections"]:
                errors.append(f"{category_config['display_name']}: Maximum {category_config['max_selections']} allowed")
            
            if category_config["required"] and not selected:
                errors.append(f"{category_config['display_name']}: This selection is required")
            
            if category_config["exclusive"] and len(selected) > 1:
                errors.append(f"{category_config['display_name']}: Only one option allowed")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def calculate_total_price(self, sector: str, item: str, selections: Dict[str, List[str]]) -> float:
        """Universal price calculation for any sector"""
        if sector not in self.sector_rules or item not in self.sector_rules[sector]:
            return 0.0
        
        base_price = self.sector_rules[sector][item].get("base_price", 0.0)
        total_price = base_price
        
        categories = self.get_selection_categories(sector, item)
        
        for category_name, selected_options in selections.items():
            if category_name in categories:
                pricing = categories[category_name]["pricing"]
                for option in selected_options:
                    total_price += pricing.get(option, 0.0)
        
        return total_price

##########################
# TRANSITION LOGIC
##########################

class GenericTransitionLogic:
    """Generic state transition logic"""
    
    def __init__(self, sector_flows: Dict[str, Dict[str, Any]]):
        self.sector_flows = sector_flows
    
    def determine_next_state(self, current_state: str, user_input: str, sector_context: str) -> str:
        if self._is_modification_request(user_input):
            return UniversalWorkflowStates.MODIFICATION
        if self._is_cancellation_request(user_input):
            return UniversalWorkflowStates.CANCELLATION
        
        sector_flow = self.sector_flows.get(sector_context, {})
        return self._get_next_required_state(current_state, sector_flow)
    
    def _is_modification_request(self, user_input: str) -> bool:
        modification_keywords = ["change", "modify", "edit", "update", "different", "alter"]
        return any(keyword in user_input.lower() for keyword in modification_keywords)
    
    def _is_cancellation_request(self, user_input: str) -> bool:
        """Enhanced cancellation detection"""
        cancellation_keywords = [
            "cancel", "quit", "exit", "stop", "abort", "nevermind", "never mind",
            "cancel this", "cancel request", "stop this", "abort this"
        ]
        user_lower = user_input.lower()
        
        # Direct keyword matching
        for keyword in cancellation_keywords:
            if keyword in user_lower:
                return True
        
        # Pattern matching for variations
        if any(phrase in user_lower for phrase in [
            "i want to cancel", "please cancel", "let's cancel", "i'll cancel"
        ]):
            return True
            
        return False


    def _get_next_required_state(self, current_state: str, sector_flow: Dict[str, Any]) -> str:
        state_order = [
            UniversalWorkflowStates.DISCOVERY,
            UniversalWorkflowStates.QUALIFICATION, 
            UniversalWorkflowStates.SELECTION,
            UniversalWorkflowStates.CUSTOMIZATION,
            UniversalWorkflowStates.INFO_COLLECTION,
            UniversalWorkflowStates.VERIFICATION,
            UniversalWorkflowStates.PAYMENT,
            UniversalWorkflowStates.FINALIZATION
        ]
        
        required_states = sector_flow.get("required_states", [])
        skip_states = sector_flow.get("skip_states", [])
        
        try:
            current_index = state_order.index(current_state)
        except ValueError:
            return UniversalWorkflowStates.FINALIZATION
        
        for next_state in state_order[current_index + 1:]:
            if next_state in required_states and next_state not in skip_states:
                return next_state
        
        return UniversalWorkflowStates.FINALIZATION

##########################
# OPENROUTER INTEGRATION (Using orderChat.py pattern)
##########################

class OpenRouterIntegration:
    """OpenRouter AI integration with proper authentication"""
    
    def __init__(self):
        # Get API key with better error handling
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            print("❌ OPENROUTER_API_KEY environment variable not found")
            self.llm_local = None
            return
        
        if not self.api_key.startswith("sk-or-v1-"):
            print(f"⚠️  API key format might be incorrect. Expected 'sk-or-v1-...', got: {self.api_key}...")
        print (f" key: {self.api_key}")
        try:
            self.llm_local = ChatOpenAI(
                model="openai/gpt-4.1",
                openai_api_key=self.api_key,
                openai_api_base=self.api_url,
                default_headers={
                    "HTTP-Referer": "https://universal-service-bot.com",  # Changed from "Referer"
                    "X-Title": "Universal Service Bot",
                    "Content-Type": "application/json"
                },
                max_retries=3,
                timeout=30  # Add timeout
            )
            print(f"✅ OpenRouter client initialized with key: {self.api_key[:10]}...")
            
        except Exception as e:
            print(f"❌ Failed to initialize OpenRouter client: {e}")
            self.llm_local = None
    
    def call_openrouter_agent(self, sector: str, user_input: str, sector_prompt: str, chat_history: list = None) -> str:
        """Call OpenRouter AI with better error handling"""
        
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
            
            # More specific error handling
            if "401" in error_msg:
                print("🔑 Authentication failed - check your OpenRouter API key")
                print(f"🔑 Current key starts with: {self.api_key[:15]}...")
            elif "rate limit" in error_msg.lower():
                print("⏰ Rate limit exceeded")
            elif "insufficient funds" in error_msg.lower():
                print("💰 Insufficient OpenRouter credits")
            
            return f"I'm here to help with {sector.replace('_', ' ')}. Could you tell me more about what you need?"


##########################
# COMPLETE UNIVERSAL SERVICE BOT
##########################

class CompleteUniversalServiceBot:
    """Complete universal service bot with file loading, OpenRouter AI, and generic rules"""
    
    def __init__(self, prompts_directory: str = "sector_prompts", rules_directory: str = "sector_rules"):
        # Load prompts and rules from files
        self.user_sector_prompts = FileLoader.load_sector_prompts_from_files(prompts_directory)
        self.sector_rules_data = FileLoader.load_sector_rules_from_files(rules_directory)
        
        # Initialize components
        self.parser = SectorPromptParser()
        self.selection_rules = GenericSelectionRules()
        self.openrouter = OpenRouterIntegration()
        
        # Load selection rules
        for sector, rules in self.sector_rules_data.items():
            self.selection_rules.load_rules_from_config(sector, rules)
        
        # Generate derived prompts
        self.derived_sector_prompts = {}
        for sector, prompt in self.user_sector_prompts.items():
            self.derived_sector_prompts[sector] = self.parser.parse_sector_prompt(prompt)
        
        # Create default sector flows
        self.sector_flows = self._create_default_sector_flows()
        
        # Initialize transition logic
        self.transition_logic = GenericTransitionLogic(self.sector_flows)
        
        # Graph cache
        self.graphs = {}
    
    def get_sector_info(self, sector: str) -> Dict[str, Any]:
        """Get information about a specific sector"""
        if sector not in self.user_sector_prompts:
            return {"error": f"Sector '{sector}' not available"}
        
        info = {
            "sector": sector,
            "has_prompt": sector in self.user_sector_prompts,
            "has_rules": sector in self.sector_rules_data,
            "customizable_items": self.selection_rules.get_customizable_items(sector),
            "workflow_states": list(self.derived_sector_prompts.get(sector, {}).keys())
        }
        
        if sector in self.sector_rules_data:
            info["rule_categories"] = list(self.sector_rules_data[sector].keys())
        
        return info

    def is_sector_available(self, sector: str) -> bool:
        """Check if a sector is available"""
        return sector in self.user_sector_prompts

    def _create_default_sector_flows2(self) -> Dict[str, Dict[str, Any]]:
        flows = {}
        for sector in self.user_sector_prompts.keys():
            if sector == "real_estate":
                # 🏠 Special optimized flow
                flows[sector] = {
                    "required_states": ["discovery", "qualification", "selection", "info_collection", "finalization"],
                    "skip_states": ["customization", "verification", "payment"],  # ✅ SKIPS problematic states
                }
            elif sector == "fitness_gym":
                # 💪 Special optimized flow  
                flows[sector] = {
                    "required_states": ["discovery", "selection", "info_collection", "payment", "finalization"],
                    "skip_states": ["customization", "verification"],  # ✅ SKIPS problematic states
                    "optional_states": [],
                    "entry_point": []
                }
            else:
                # ❌ DEFAULT FLOW - includes ALL 8 states!
                flows[sector] = {
                    "required_states": ["discovery", "selection", "customization", "info_collection", "payment", "finalization"],
                    "optional_states": ["qualification", "verification"],
                    "skip_states": [],  # ❌ NO SKIPPED STATES
                    "entry_point": []
                }


    def _create_default_sector_flows(self) -> Dict[str, Dict[str, Any]]:
        """Create default sector flows for all loaded sectors"""
        flows = {}
        for sector in self.user_sector_prompts.keys():
            if sector == "real_estate":
                # 🏠 Special flow for real estate to prevent loops
                flows[sector] = {
                    "required_states": ["discovery", "qualification", "selection", "info_collection", "finalization"],
                    "optional_states": [],
                    "skip_states": ["customization", "verification", "payment"],  # Skip problematic states
                    "entry_point": "discovery"
                }
            elif sector == "fitness_gym":
                # 💪 Special flow for fitness to prevent loops
                flows[sector] = {
                    "required_states": ["discovery", "selection", "info_collection", "payment", "finalization"],
                    "optional_states": ["qualification"],
                    "skip_states": ["customization", "verification"],  # Skip problematic states
                    "entry_point": "discovery"
                }
            else:
                flows[sector] = {
                    "required_states": ["discovery", "selection", "customization", "info_collection", "payment", "finalization"],
                    "optional_states": ["qualification", "verification"],
                    "skip_states": [],
                    "entry_point": "discovery"
                }
        return flows


    def create_state_graph(self, sector: str) -> StateGraph:
        """Create state graph for a specific sector"""
        if sector not in self.derived_sector_prompts:
            raise ValueError(f"Sector '{sector}' not found")
        
        if sector in self.graphs:
            return self.graphs[sector]
        
        all_states = [
            UniversalWorkflowStates.DISCOVERY,
            UniversalWorkflowStates.QUALIFICATION,
            UniversalWorkflowStates.SELECTION,
            UniversalWorkflowStates.CUSTOMIZATION,
            UniversalWorkflowStates.INFO_COLLECTION,
            UniversalWorkflowStates.VERIFICATION,
            UniversalWorkflowStates.PAYMENT,
            UniversalWorkflowStates.FINALIZATION,
            UniversalWorkflowStates.MODIFICATION,
            UniversalWorkflowStates.CANCELLATION
        ]
        
        graph_builder = StateGraph(UniversalState)
        
        for state in all_states:
            graph_builder.add_node(state, self._create_handler(sector, state))
        
        entry_point = self.sector_flows[sector].get("entry_point", UniversalWorkflowStates.DISCOVERY)
        # ✅ Alternative - explicit None check
        entry_point = self.sector_flows[sector].get("entry_point", UniversalWorkflowStates.DISCOVERY)         
        graph_builder.add_edge(START, entry_point)
        
        workflow_states = all_states[:-2]
        
        for current_state in workflow_states:
            if current_state == UniversalWorkflowStates.FINALIZATION:
                graph_builder.add_conditional_edges(
                    current_state,
                    lambda s: self._route_final_state(s),
                    {
                        "end": END,
                        "modification": UniversalWorkflowStates.MODIFICATION,
                        "cancellation": UniversalWorkflowStates.CANCELLATION
                    }
                )
            else:
                next_state = self._get_routing_target(current_state, sector)
                graph_builder.add_conditional_edges(
                    current_state,
                    lambda s, curr=current_state: self._route_from_state(s, curr, sector),
                    {
                        "next": next_state,
                        "modification": UniversalWorkflowStates.MODIFICATION,
                        "cancellation": UniversalWorkflowStates.CANCELLATION
                    }
                )
        
        modification_targets = {state: state for state in workflow_states}
        graph_builder.add_conditional_edges(
            UniversalWorkflowStates.MODIFICATION,
            lambda s: self._route_from_modification(s),
            modification_targets
        )
        
        graph_builder.add_edge(UniversalWorkflowStates.CANCELLATION, END)
        
        compiled_graph = graph_builder.compile()
        self.graphs[sector] = compiled_graph
        
        return compiled_graph
    
    def _get_routing_target(self, current_state: str, sector: str) -> str:
        return self.transition_logic._get_next_required_state(current_state, self.sector_flows[sector])
    
    
    def _create_handler(self, sector: str, state: str):
        def handler(current_state: UniversalState) -> Dict[str, Any]:
            user_input = ""
            if current_state["messages"]:
                user_input = current_state["messages"][-1].content
            
            session_data = current_state.get("session_data", {})
            step_count = session_data.get("step_count", 0)
            
            # 🔧 FIX: Only call AI for NEW user input, not state transitions
            if user_input and len(user_input.strip()) > 0:
                # User provided new input - make ONE smart AI call
                full_sector_prompt = self.user_sector_prompts[sector]  # Use FULL prompt, not state fragments
                chat_history = current_state.get("chat_history", [])
                ai_response = self.openrouter.call_openrouter_agent(sector, user_input, full_sector_prompt, chat_history)
                
                # Let AI be smart - it can handle the full conversation flow
                return {
                    "messages": [AIMessage(content=ai_response)],
                    "current_step": UniversalWorkflowStates.FINALIZATION,  # AI handles everything
                    "completed": True,  # One intelligent response completes the flow
                    "session_data": {**session_data, "step_count": step_count + 1}
                }
            else:
                # No new user input - use static response for internal transitions
                return {
                    "messages": [AIMessage(content="Processing your request...")],
                    "current_step": state,
                    "completed": True
                }
        return handler


    def _create_handler2(self, sector: str, state: str):
        """Create handler with 100% AI responses - no static responses"""
        def handler(current_state: UniversalState) -> Dict[str, Any]:
            # Get user input
            user_input = ""
            if current_state["messages"]:
                user_input = current_state["messages"][-1].content
            
            # Session data and step counting
            session_data = current_state.get("session_data", {})
            step_count = session_data.get("step_count", 0)
            
            # Force completion if too many steps
            max_steps = 20 if sector in ["fitness_gym", "real_estate"] else 25
            if step_count > max_steps:
                return {
                    "messages": [AIMessage(content=f"Your {sector.replace('_', ' ')} request is complete! Thank you.")],
                    "current_step": UniversalWorkflowStates.FINALIZATION,
                    "completed": True,
                    "session_data": {**session_data, "step_count": step_count + 1}
                }
            
            # 🤖 ALWAYS call AI for ALL states - no static responses
            sector_prompt = self.derived_sector_prompts[sector].get(state, f"How can I help you with {sector.replace('_', ' ')}?")
            chat_history = current_state.get("chat_history", [])
            ai_response = self.openrouter.call_openrouter_agent(sector, user_input, sector_prompt, chat_history)
            
            updates = {
                "messages": [AIMessage(content=ai_response)],
                "current_step": state,
                "session_data": {
                    **session_data, 
                    f"{state}_completed": True,
                    "last_user_input": user_input,
                    "ai_response": ai_response,
                    "step_count": step_count + 1
                }
            }
            
            # Enhanced selection and customization with rules
            if state == UniversalWorkflowStates.SELECTION:
                return self._handle_rule_based_selection(current_state, updates, sector)
            elif state == UniversalWorkflowStates.CUSTOMIZATION:
                return self._handle_rule_based_customization(current_state, updates, sector)
            elif state == UniversalWorkflowStates.QUALIFICATION:
                updates["qualified"] = True
            elif state == UniversalWorkflowStates.PAYMENT:
                updates["payment_processed"] = True
                total = self._calculate_simple_total(current_state.get("selected_items", []))
                updates["session_data"]["total_amount"] = total
            elif state == UniversalWorkflowStates.FINALIZATION:
                updates["completed"] = True
            elif state == UniversalWorkflowStates.MODIFICATION:
                updates["session_data"]["return_to_state"] = current_state.get("current_step", UniversalWorkflowStates.SELECTION)
            elif state == UniversalWorkflowStates.CANCELLATION:
                updates["completed"] = True
            
            # Special handling for problematic sectors
            if sector in ["real_estate", "fitness_gym"] and state == UniversalWorkflowStates.INFO_COLLECTION:
                updates["completed"] = True
                updates["current_step"] = UniversalWorkflowStates.FINALIZATION
            
            return updates
        
        return handler


    def _handle_rule_based_selection(self, current_state: UniversalState, base_response: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """Handle selection with rule-based customizable items"""
        user_input = current_state["messages"][-1].content.lower() if current_state["messages"] else ""
        
        customizable_items = self.selection_rules.get_customizable_items(sector)
        selected_item = None
        
        for item in customizable_items:
            if item.lower() in user_input:
                selected_item = item
                break
        
        if selected_item:
            categories = self.selection_rules.get_selection_categories(sector, selected_item)
            
            if categories:
                customization_prompt = f"Perfect! Let's customize your {selected_item}.\n\n"
                
                for category_name, category_info in categories.items():
                    customization_prompt += f"**{category_info['display_name']}**\n"
                    
                    for option in category_info['options']:
                        price_info = ""
                        if option in category_info['pricing'] and category_info['pricing'][option] != 0:
                            price_info = f" (+${category_info['pricing'][option]:.2f})"
                        customization_prompt += f"• {option}{price_info}\n"
                    
                    if category_info['required']:
                        min_sel = category_info['min_selections']
                        max_sel = category_info['max_selections']
                        if category_info['exclusive']:
                            customization_prompt += f"(Required: select exactly 1)\n"
                        else:
                            customization_prompt += f"(Required: select {min_sel}-{max_sel})\n"
                    else:
                        customization_prompt += f"(Optional: up to {category_info['max_selections']})\n"
                    customization_prompt += "\n"
                
                base_response["messages"] = [AIMessage(content=customization_prompt)]
                base_response["session_data"]["customizing_item"] = selected_item
                base_response["session_data"]["available_categories"] = categories
                base_response["session_data"]["current_selections"] = {}
        
        return base_response
    
    def _handle_rule_based_customization(self, current_state: UniversalState, base_response: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """Handle customization with validation"""
        user_input = current_state["messages"][-1].content if current_state["messages"] else ""
        
        customizing_item = current_state["session_data"].get("customizing_item")
        current_selections = current_state["session_data"].get("current_selections", {})
        
        if customizing_item:
            new_selections = self._parse_customization_input(user_input, sector, customizing_item)
            current_selections.update(new_selections)
            
            validation = self.selection_rules.validate_selections(sector, customizing_item, current_selections)
            
            if validation["valid"]:
                total_price = self.selection_rules.calculate_total_price(sector, customizing_item, current_selections)
                
                confirmation = f"Excellent! Your customized {customizing_item}:\n\n"
                for category, items in current_selections.items():
                    if items:
                        categories = self.selection_rules.get_selection_categories(sector, customizing_item)
                        display_name = categories.get(category, {}).get('display_name', category.title())
                        confirmation += f"**{display_name}:** {', '.join(items)}\n"
                
                if total_price > 0:
                    confirmation += f"\n**Total Price:** ${total_price:.2f}\n"
                confirmation += "\nWould you like to proceed with this selection?"
                
                base_response["messages"] = [AIMessage(content=confirmation)]
                base_response["session_data"]["customization_complete"] = True
                base_response["session_data"]["final_selections"] = current_selections
                base_response["session_data"]["final_price"] = total_price
            else:
                error_message = "Please make the following corrections:\n\n"
                for error in validation["errors"]:
                    error_message += f"• {error}\n"
                error_message += "\nWhat would you like to select?"
                
                base_response["messages"] = [AIMessage(content=error_message)]
        
        return base_response
    
    def _parse_customization_input(self, user_input: str, sector: str, item: str) -> Dict[str, List[str]]:
        """Parse user input for customization selections"""
        selections = {}
        categories = self.selection_rules.get_selection_categories(sector, item)
        
        user_lower = user_input.lower()
        
        for category_name, category_info in categories.items():
            category_selections = []
            for option in category_info["options"]:
                if option.lower() in user_lower:
                    category_selections.append(option)
            
            if category_selections:
                selections[category_name] = category_selections
        
        return selections
    
    def _calculate_simple_total(self, selected_items: List[str]) -> float:
        """Simple price calculation from selected items"""
        total = 0.0
        for item in selected_items:
            price_match = re.search(r'\$(\d+(?:\.\d{2})?)', item)
            if price_match:
                total += float(price_match.group(1))
        return total
    
    def _route_from_state(self, state: UniversalState, current_state: str, sector: str) -> str:
        """Enhanced routing logic with better keyword detection"""
        if not state["messages"]:
            return "next"
        
        user_input = state["messages"][-1].content.lower()
        
        # Enhanced cancellation detection
        if any(word in user_input for word in ["cancel", "quit", "exit", "stop", "abort"]):
            return "cancellation"
        
        # Enhanced modification detection  
        if any(word in user_input for word in ["change", "modify", "edit", "different", "alter"]) and not any(word in user_input for word in ["charge", "card", "process", "payment"]):
            return "modification"
        
        # Default progression
        return "next"


    def _route_final_state(self, state: UniversalState) -> str:
        if not state["messages"]:
            return "end"
        
        user_input = state["messages"][-1].content.lower()
        
        if any(word in user_input for word in ["modify", "change", "edit"]):
            return "modification"
        if any(word in user_input for word in ["cancel", "quit", "exit"]):
            return "cancellation"
        
        return "end"
    
    def _route_from_modification(self, state: UniversalState) -> str:
        return state["session_data"].get("return_to_state", UniversalWorkflowStates.SELECTION)
    
    def get_loaded_sectors(self) -> List[str]:
        """Get list of successfully loaded sectors"""
        return list(self.user_sector_prompts.keys())
    
    def get_derived_prompts(self, sector: str) -> Dict[str, str]:
        """Get derived prompts for debugging"""
        return self.derived_sector_prompts.get(sector, {})
    
    def chat_away_universal(self, sector: str, user_input: str, chat_history: list = None) -> str:
        """One intelligent AI call handles the entire conversation"""
        try:
            # Use FULL sector prompt - let AI be intelligent
            sector_prompt = self.user_sector_prompts.get(sector, f"Welcome! How can I help you with {sector.replace('_', ' ')}?")
            
            # ONE smart AI call handles everything
            response = self.openrouter.call_openrouter_agent(sector, user_input, sector_prompt, chat_history)
            
            return response
        except Exception as e:
            return "An error occurred. Please try again."


    def chat_away_universal2(self, sector: str, user_input: str, chat_history: list = None) -> str:
        """Universal version of orderChat.py chatAway method"""
        try:
            # Get sector prompt
            sector_prompt = self.user_sector_prompts.get(sector, f"Welcome! How can I help you with {sector.replace('_', ' ')}?")
            
            # Call OpenRouter exactly like orderChat.py
            response = self.openrouter.call_openrouter_agent(sector, user_input, sector_prompt, chat_history)
            return response
            
        except Exception as e:
            print(f"Universal workflow error: {str(e)}")
            return "An error occurred. Please start over."

##########################
# EXAMPLE USAGE
##########################

def main():
    """Example usage of the complete universal service bot"""
    print("🤖 Complete Universal Service Bot with OpenRouter AI")
    print("=" * 60)
    
    # Initialize the complete bot (reads from existing sector_prompts folder)
    bot = CompleteUniversalServiceBot()
    
    # Show loaded sectors
    loaded_sectors = bot.get_loaded_sectors()
    print(f"✅ Loaded {len(loaded_sectors)} sectors: {', '.join(loaded_sectors)}")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n⚠️ Set OPENROUTER_API_KEY environment variable for AI integration")
        print("export OPENROUTER_API_KEY='your_key_here'")
    
    # Interactive chat example
    if loaded_sectors:
        print(f"\n🗣️ Try chatting with the bot:")
        print("Example: 'Hello, I need help with healthcare'")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                # Simple sector detection
                detected_sector = None
                for sector in loaded_sectors:
                    if sector.replace('_', ' ') in user_input.lower():
                        detected_sector = sector
                        break
                
                if not detected_sector:
                    detected_sector = loaded_sectors[0]  # Default to first sector
                
                # Get AI response
                response = bot.chat_away_universal(detected_sector, user_input)
                print(f"Bot ({detected_sector}): {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

if __name__ == "__main__":
    main()

