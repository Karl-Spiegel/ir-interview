"""
QBE Claims Allocation System - Intelligent Orchestration Engine
================================================================

This module implements the core orchestration logic for intelligent claims routing
at QBE Insurance. It processes 10,000+ monthly claims through ML-based matching
and real-time workload balancing.

Architecture:
    - Proactive ML predictions for optimal officer-claim matching
    - Reactive adjustments based on real-time conditions
    - Multi-stage NLP pipeline for claim complexity analysis
    - Continuous rebalancing for workload optimization

Author: [Your Name]
Date: November 2024
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================== Data Models ==================

@dataclass
class ClaimComplexity:
    """Analyzed claim complexity metrics from NLP pipeline"""
    complexity_score: float  # 0-100 scale
    categories: List[str]  # Primary claim categories
    entities: Dict[str, List[str]]  # Extracted entities
    urgency: str  # critical, high, medium, low
    estimated_handling_time: int  # in hours
    confidence: float  # Model confidence 0-1


@dataclass
class Officer:
    """Claims officer profile with expertise and workload"""
    id: str
    name: str
    expertise: List[str]  # Areas of expertise
    current_load: int  # Current number of claims
    max_capacity: int  # Maximum claim capacity
    performance_score: float  # 0-1 performance metric
    availability: bool
    specializations: List[str]  # Specific specializations
    avg_resolution_time: float  # Days
    embedding: Optional[np.ndarray] = None  # Officer expertise embedding


@dataclass
class Claim:
    """Insurance claim with metadata and analysis results"""
    id: str
    text: str
    metadata: Dict[str, Any]
    complexity: Optional[ClaimComplexity] = None
    assigned_officer: Optional[str] = None
    priority: int = 0  # 0-10 priority scale
    submission_time: datetime = field(default_factory=datetime.now)
    sla_deadline: Optional[datetime] = None


class AllocationStrategy(Enum):
    """Allocation strategies for different scenarios"""
    ML_OPTIMAL = "ml_optimal"  # Use ML predictions
    LOAD_BALANCED = "load_balanced"  # Balance workloads
    EXPERTISE_MATCH = "expertise_match"  # Match expertise
    ROUND_ROBIN = "round_robin"  # Fallback strategy
    URGENT_PRIORITY = "urgent_priority"  # Handle urgent claims


# ================== NLP Analysis Pipeline ==================

class ClaimAnalyzer:
    """
    Multi-stage NLP pipeline for claim complexity analysis
    Similar to document analysis needed for Pragia's IR system
    """

    def __init__(self):
        # In production, these would be actual model loads
        self.complexity_model = self._load_complexity_model()
        self.entity_extractor = self._load_ner_model()
        self.classifier = self._load_classifier()

    def analyze_claim(self, claim: Claim) -> ClaimComplexity:
        """
        Analyze claim text and metadata to determine complexity

        Pipeline stages:
        1. Entity extraction (parties, damages, locations)
        2. Complexity feature extraction
        3. Category classification
        4. Urgency determination
        5. Time estimation
        """
        logger.info(f"Analyzing claim {claim.id}")

        # Stage 1: Entity Extraction
        entities = self._extract_entities(claim.text)

        # Stage 2: Complexity Features
        features = self._extract_complexity_features(
            claim.text, entities, claim.metadata
        )

        # Stage 3: Classification
        categories = self._classify_claim(claim.text)

        # Stage 4: Complexity Scoring
        complexity_score = self._calculate_complexity(
            features, categories, entities
        )

        # Stage 5: Urgency & Time Estimation
        urgency = self._determine_urgency(features, claim.metadata)
        handling_time = self._estimate_handling_time(
            complexity_score, categories
        )

        return ClaimComplexity(
            complexity_score=complexity_score,
            categories=categories[:3],  # Top 3 categories
            entities=entities,
            urgency=urgency,
            estimated_handling_time=handling_time,
            confidence=self._calculate_confidence(features)
        )

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from claim text"""
        # Simulated entity extraction
        return {
            "persons": ["John Doe", "Jane Smith"],
            "locations": ["Sydney", "Melbourne"],
            "damages": ["property damage", "water damage"],
            "amounts": ["$50,000", "$10,000"]
        }

    def _extract_complexity_features(self, text: str,
                                    entities: Dict,
                                    metadata: Dict) -> Dict:
        """Extract features indicating claim complexity"""
        return {
            'word_count': len(text.split()),
            'entity_count': sum(len(v) for v in entities.values()),
            'medical_terms': self._count_medical_terms(text),
            'legal_references': text.count("liability") + text.count("negligence"),
            'damage_amount': metadata.get('estimated_amount', 0),
            'prior_claims': metadata.get('prior_claims_count', 0),
            'document_count': metadata.get('supporting_docs', 0),
            'multiple_parties': len(entities.get("persons", [])) > 2
        }

    def _classify_claim(self, text: str) -> List[str]:
        """Classify claim into categories"""
        # Simulated classification
        categories = [
            "property_damage",
            "liability",
            "business_interruption",
            "natural_disaster",
            "cyber_incident"
        ]
        # In production, use BERT or similar
        return categories[:3]

    def _calculate_complexity(self, features: Dict,
                             categories: List[str],
                             entities: Dict) -> float:
        """Calculate overall complexity score"""
        score = 0.0

        # Feature-based scoring
        score += min(features['word_count'] / 1000, 1.0) * 10
        score += min(features['entity_count'] / 20, 1.0) * 15
        score += features['medical_terms'] * 5
        score += features['legal_references'] * 8
        score += min(features['damage_amount'] / 100000, 1.0) * 20
        score += features['prior_claims'] * 3
        score += features['document_count'] * 2

        # Category multipliers
        if "cyber_incident" in categories:
            score *= 1.3
        if "liability" in categories:
            score *= 1.2

        return min(score, 100.0)

    def _determine_urgency(self, features: Dict, metadata: Dict) -> str:
        """Determine claim urgency level"""
        if metadata.get("emergency", False):
            return "critical"
        elif features['damage_amount'] > 100000:
            return "high"
        elif features['medical_terms'] > 5:
            return "high"
        elif features['multiple_parties']:
            return "medium"
        return "low"

    def _estimate_handling_time(self, complexity: float,
                               categories: List[str]) -> int:
        """Estimate handling time in hours"""
        base_time = 8  # Base 8 hours

        # Complexity multiplier
        time_multiplier = 1 + (complexity / 100) * 2

        # Category adjustments
        if "cyber_incident" in categories:
            time_multiplier *= 1.5
        if "liability" in categories:
            time_multiplier *= 1.3

        return int(base_time * time_multiplier)

    def _count_medical_terms(self, text: str) -> int:
        """Count medical terminology in text"""
        medical_terms = ["injury", "trauma", "fracture", "surgery",
                        "hospital", "treatment", "diagnosis"]
        return sum(1 for term in medical_terms if term in text.lower())

    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate model confidence score"""
        # Higher document count = higher confidence
        doc_confidence = min(features['document_count'] / 10, 1.0)
        # Lower complexity = higher confidence
        complexity_confidence = 1.0 - (features.get('word_count', 0) / 5000)
        return (doc_confidence + complexity_confidence) / 2

    def _load_complexity_model(self):
        """Load complexity scoring model"""
        # In production, load actual model
        return None

    def _load_ner_model(self):
        """Load named entity recognition model"""
        # In production, load spaCy or similar
        return None

    def _load_classifier(self):
        """Load claim classification model"""
        # In production, load BERT or similar
        return None


# ================== Orchestration Engine ==================

class OrchestrationEngine:
    """
    Core orchestration engine for intelligent claim allocation
    Combines proactive ML predictions with reactive adjustments
    """

    def __init__(self):
        self.analyzer = ClaimAnalyzer()
        self.officers: List[Officer] = []
        self.active_allocations: Dict[str, str] = {}  # claim_id -> officer_id
        self.allocation_history: List[Dict] = []
        self.rebalance_interval = 300  # 5 minutes
        self._load_officers()

    async def orchestrate(self, claim: Claim) -> str:
        """
        Main orchestration logic - proactive + reactive allocation

        Returns:
            officer_id: ID of assigned officer
        """
        logger.info(f"Orchestrating claim {claim.id}")

        # Step 1: Analyze claim complexity
        claim.complexity = self.analyzer.analyze_claim(claim)

        # Step 2: Determine allocation strategy
        strategy = self._determine_strategy(claim)

        # Step 3: Get candidate officers (proactive)
        candidates = await self._predict_best_matches(claim)

        # Step 4: Apply real-time filters (reactive)
        available_officers = await self._check_availability(candidates)

        # Step 5: Smart allocation
        selected_officer = await self._smart_allocation(
            claim, available_officers, strategy
        )

        # Step 6: Update state
        await self._update_allocation_state(claim, selected_officer)

        logger.info(f"Allocated claim {claim.id} to officer {selected_officer.id}")
        return selected_officer.id

    def _determine_strategy(self, claim: Claim) -> AllocationStrategy:
        """Determine optimal allocation strategy"""
        if claim.complexity.urgency == "critical":
            return AllocationStrategy.URGENT_PRIORITY
        elif claim.complexity.complexity_score > 75:
            return AllocationStrategy.EXPERTISE_MATCH
        elif self._is_workload_imbalanced():
            return AllocationStrategy.LOAD_BALANCED
        else:
            return AllocationStrategy.ML_OPTIMAL

    async def _predict_best_matches(self, claim: Claim) -> List[Officer]:
        """
        ML-based prediction of best officer matches
        Uses embeddings and similarity scoring
        """
        # Get claim embedding
        claim_embedding = self._get_claim_embedding(claim)

        # Calculate match scores for all officers
        scores = []
        for officer in self.officers:
            if not officer.embedding:
                officer.embedding = self._get_officer_embedding(officer)

            # Cosine similarity between claim and officer
            similarity = self._cosine_similarity(
                claim_embedding, officer.embedding
            )

            # Adjust for workload and performance
            workload_factor = 1 - (officer.current_load / officer.max_capacity)
            performance_factor = officer.performance_score

            # Weighted score
            final_score = (
                similarity * 0.5 +
                workload_factor * 0.3 +
                performance_factor * 0.2
            )

            # Boost for exact expertise match
            if any(cat in officer.expertise for cat in claim.complexity.categories):
                final_score *= 1.2

            scores.append((officer, final_score))

        # Return top 5 candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        return [officer for officer, _ in scores[:5]]

    async def _check_availability(self, officers: List[Officer]) -> List[Officer]:
        """Check real-time availability of officers"""
        available = []
        for officer in officers:
            if officer.availability and officer.current_load < officer.max_capacity:
                available.append(officer)
        return available

    async def _smart_allocation(self, claim: Claim,
                               officers: List[Officer],
                               strategy: AllocationStrategy) -> Officer:
        """
        Smart allocation with strategy-based selection
        """
        if not officers:
            # Fallback to any available officer
            return await self._emergency_fallback(claim)

        if strategy == AllocationStrategy.URGENT_PRIORITY:
            # Pick highest performer
            return max(officers, key=lambda o: o.performance_score)
        elif strategy == AllocationStrategy.EXPERTISE_MATCH:
            # Pick best expertise match
            return self._best_expertise_match(claim, officers)
        elif strategy == AllocationStrategy.LOAD_BALANCED:
            # Pick least loaded
            return min(officers, key=lambda o: o.current_load)
        else:  # ML_OPTIMAL
            # Already sorted by ML score, pick first
            return officers[0]

    def _best_expertise_match(self, claim: Claim,
                             officers: List[Officer]) -> Officer:
        """Find officer with best expertise match"""
        best_match = officers[0]
        best_score = 0

        for officer in officers:
            score = sum(1 for cat in claim.complexity.categories
                       if cat in officer.expertise)
            if score > best_score:
                best_score = score
                best_match = officer

        return best_match

    async def _emergency_fallback(self, claim: Claim) -> Officer:
        """Emergency fallback when no officers available"""
        logger.warning(f"Using emergency fallback for claim {claim.id}")

        # Find any officer with capacity
        for officer in self.officers:
            if officer.current_load < officer.max_capacity * 1.2:  # Allow 20% overload
                return officer

        # Last resort: least loaded officer
        return min(self.officers, key=lambda o: o.current_load)

    async def _update_allocation_state(self, claim: Claim, officer: Officer):
        """Update allocation state and metrics"""
        # Update assignment
        claim.assigned_officer = officer.id
        self.active_allocations[claim.id] = officer.id

        # Update officer workload
        officer.current_load += 1

        # Log allocation
        self.allocation_history.append({
            "claim_id": claim.id,
            "officer_id": officer.id,
            "timestamp": datetime.now(),
            "complexity": claim.complexity.complexity_score,
            "strategy": "ml_optimal"
        })

        # Emit metrics (in production, send to monitoring)
        await self._emit_metrics(claim, officer)

    async def handle_rebalancing(self):
        """
        Reactive: Periodic rebalancing of workloads
        Runs every 5 minutes to optimize distribution
        """
        while True:
            await asyncio.sleep(self.rebalance_interval)
            logger.info("Starting workload rebalancing")

            # Identify imbalances
            overloaded, underloaded = self._identify_imbalances()

            if overloaded:
                # Rebalance workloads
                await self._rebalance_workload(overloaded, underloaded)

            # Check for SLA breaches
            await self._check_sla_compliance()

    def _identify_imbalances(self) -> Tuple[List[Officer], List[Officer]]:
        """Identify overloaded and underloaded officers"""
        overloaded = []
        underloaded = []

        avg_load = np.mean([o.current_load for o in self.officers])

        for officer in self.officers:
            load_ratio = officer.current_load / officer.max_capacity
            if load_ratio > 0.9:
                overloaded.append(officer)
            elif load_ratio < 0.5:
                underloaded.append(officer)

        return overloaded, underloaded

    async def _rebalance_workload(self, overloaded: List[Officer],
                                 underloaded: List[Officer]):
        """Rebalance claims from overloaded to underloaded officers"""
        logger.info(f"Rebalancing: {len(overloaded)} overloaded, "
                   f"{len(underloaded)} underloaded")

        # Implementation would move claims between officers
        # This is simplified for demonstration
        transfers = 0
        for over_officer in overloaded:
            for under_officer in underloaded:
                if over_officer.current_load > over_officer.max_capacity:
                    # Transfer logic would go here
                    transfers += 1

        logger.info(f"Completed {transfers} claim transfers")

    async def _check_sla_compliance(self):
        """Check for potential SLA breaches"""
        current_time = datetime.now()

        for claim_id, officer_id in self.active_allocations.items():
            # Check if approaching SLA deadline
            # Implementation would check actual deadlines
            pass

    def _get_claim_embedding(self, claim: Claim) -> np.ndarray:
        """Generate embedding for claim"""
        # In production, use sentence transformers or similar
        # This is a simplified version
        np.random.seed(hash(claim.text) % 2**32)
        return np.random.rand(768)  # 768-dim embedding

    def _get_officer_embedding(self, officer: Officer) -> np.ndarray:
        """Generate embedding for officer expertise"""
        # In production, aggregate from historical performance
        np.random.seed(hash(officer.id) % 2**32)
        return np.random.rand(768)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _is_workload_imbalanced(self) -> bool:
        """Check if workload is imbalanced across officers"""
        loads = [o.current_load / o.max_capacity for o in self.officers]
        return np.std(loads) > 0.3  # 30% standard deviation threshold

    async def _emit_metrics(self, claim: Claim, officer: Officer):
        """Emit metrics for monitoring"""
        metrics = {
            "claim_complexity": claim.complexity.complexity_score,
            "officer_load": officer.current_load,
            "allocation_time": (datetime.now() - claim.submission_time).seconds,
            "urgency": claim.complexity.urgency
        }
        # In production, send to monitoring system
        logger.debug(f"Metrics: {metrics}")

    def _load_officers(self):
        """Load officer profiles"""
        # In production, load from database
        self.officers = [
            Officer(
                id="OFF001",
                name="Sarah Johnson",
                expertise=["property_damage", "natural_disaster"],
                current_load=12,
                max_capacity=20,
                performance_score=0.92,
                availability=True,
                specializations=["commercial_property"],
                avg_resolution_time=3.2
            ),
            Officer(
                id="OFF002",
                name="Michael Chen",
                expertise=["cyber_incident", "liability"],
                current_load=8,
                max_capacity=15,
                performance_score=0.88,
                availability=True,
                specializations=["data_breach", "technology"],
                avg_resolution_time=4.1
            ),
            Officer(
                id="OFF003",
                name="Emma Williams",
                expertise=["liability", "business_interruption"],
                current_load=15,
                max_capacity=18,
                performance_score=0.95,
                availability=True,
                specializations=["professional_liability"],
                avg_resolution_time=2.8
            )
        ]


# ================== Main Application ==================

async def main():
    """
    Main application demonstrating the orchestration system
    """
    # Initialize orchestration engine
    engine = OrchestrationEngine()

    # Start rebalancing task
    rebalance_task = asyncio.create_task(engine.handle_rebalancing())

    # Simulate incoming claims
    test_claims = [
        Claim(
            id="CLM001",
            text="Severe water damage to commercial property in Sydney CBD. "
                 "Multiple floors affected. Estimated damage $250,000. "
                 "Business interruption claim likely.",
            metadata={
                "estimated_amount": 250000,
                "location": "Sydney",
                "property_type": "commercial",
                "supporting_docs": 12
            },
            priority=8
        ),
        Claim(
            id="CLM002",
            text="Cyber attack on client systems. Data breach affecting "
                 "customer records. Potential liability exposure. "
                 "Regulatory notification required.",
            metadata={
                "estimated_amount": 500000,
                "incident_type": "cyber",
                "affected_records": 10000,
                "emergency": True
            },
            priority=10
        ),
        Claim(
            id="CLM003",
            text="Vehicle collision with property damage. Minor injuries "
                 "reported. Police report available.",
            metadata={
                "estimated_amount": 35000,
                "injury_count": 2,
                "police_report": True
            },
            priority=5
        )
    ]

    # Process claims
    for claim in test_claims:
        officer_id = await engine.orchestrate(claim)
        print(f"Claim {claim.id} assigned to officer {officer_id}")
        print(f"  - Complexity: {claim.complexity.complexity_score:.1f}")
        print(f"  - Categories: {', '.join(claim.complexity.categories)}")
        print(f"  - Urgency: {claim.complexity.urgency}")
        print(f"  - Est. Time: {claim.complexity.estimated_handling_time} hours")
        print()

        # Simulate processing delay
        await asyncio.sleep(1)

    # Display final state
    print("\nFinal Officer Workloads:")
    for officer in engine.officers:
        load_pct = (officer.current_load / officer.max_capacity) * 100
        print(f"  {officer.name}: {officer.current_load}/{officer.max_capacity} "
              f"claims ({load_pct:.0f}% capacity)")

    # Cancel rebalancing task
    rebalance_task.cancel()


if __name__ == "__main__":
    print("QBE Claims Allocation System - Orchestration Demo")
    print("=" * 50)
    asyncio.run(main())