# Project 2: QBE Claims Intelligent Allocation System
## QBE Insurance (2023)

### Executive Summary
Designed and implemented an ML-driven claims allocation system processing **10,000+ claims monthly** with real-time complexity assessment and intelligent routing.

### Problem Statement
- Manual claims allocation led to inefficient workload distribution
- Senior adjusters overwhelmed with simple claims
- Complex claims misrouted to junior staff causing delays
- No visibility into claim complexity before allocation

### Solution Architecture

```mermaid
graph LR
    A[Incoming Claim] --> B[NLP Pipeline]
    B --> C[Complexity Scorer]
    C --> D[Skills Matcher]
    D --> E[Workload Balancer]
    E --> F[Allocation Engine]
    F --> G[Assignment + Notification]

    H[ML Training Pipeline]
    H --> C
    H --> D

    I[Feedback Loop]
    G --> I
    I --> H
```

### Technical Implementation

```python
class ClaimsAllocationSystem:
    def __init__(self):
        self.complexity_model = self._load_complexity_model()
        self.skills_matcher = SkillsMatchingEngine()
        self.workload_tracker = WorkloadBalancer()

    async def process_claim(self, claim: ClaimDocument):
        # Extract features using NLP
        features = await self.nlp_pipeline.extract_features(claim)

        # Score complexity (0-10 scale)
        complexity = self.complexity_model.predict(features)

        # Find eligible adjusters based on skills
        eligible_adjusters = self.skills_matcher.match(
            claim_type=claim.type,
            complexity=complexity,
            required_certifications=claim.certifications
        )

        # Balance workload
        selected_adjuster = self.workload_tracker.select_optimal(
            eligible_adjusters,
            complexity_weight=complexity
        )

        return AllocationDecision(
            adjuster=selected_adjuster,
            complexity_score=complexity,
            confidence=features.confidence,
            reasoning=self._generate_reasoning(features, selected_adjuster)
        )
```

### ML Pipeline Components

1. **NLP Feature Extraction**
   - Named Entity Recognition for policy details
   - Sentiment analysis for urgency detection
   - Document classification for claim type
   - RegEx patterns for specific compliance flags

2. **Complexity Scoring Model**
   - XGBoost ensemble with 47 features
   - Trained on 2 years of historical data
   - Monthly retraining with human feedback
   - A/B testing framework for model updates

3. **Skills Matching Algorithm**
   - Graph-based matching of adjuster skills to claim requirements
   - Dynamic skill proficiency updates based on performance
   - Certification tracking and compliance validation

### Production Architecture

```yaml
Infrastructure:
  API:
    - FastAPI microservices
    - Redis for caching
    - RabbitMQ for async processing

  ML Serving:
    - Model registry with versioning
    - Blue-green deployments
    - Shadow mode testing
    - Feature store (Feast)

  Monitoring:
    - Prometheus metrics
    - Grafana dashboards
    - Custom drift detection
    - Performance tracking per adjuster
```

### Impact Metrics

- **35% reduction** in average claim processing time
- **50% improvement** in first-touch resolution
- **$2.3M annual savings** from efficiency gains
- **92% adjuster satisfaction** with allocations
- **99.7% uptime** over 12 months

### Proactive + Reactive Features

**Proactive:**
- Predicts claim complexity before human review
- Suggests similar historical claims for reference
- Alerts on unusual patterns or potential fraud

**Reactive:**
- Real-time reallocation on adjuster unavailability
- Escalation workflows for misallocated claims
- Feedback incorporation within 24 hours

### Connection to Pragia/IRIS
This demonstrates:
- **Production ML at scale**: Handling real-time decisions with business impact
- **Orchestration expertise**: Complex workflow management similar to agent systems
- **Observability**: Complete monitoring and feedback loops
- **Human-in-the-loop ML**: Balancing automation with expert judgment

### Technical Challenges Overcome
1. **Cold start problem**: Bootstrapped with rule-based system, gradually introduced ML
2. **Class imbalance**: Only 5% of claims were "complex" - used SMOTE and custom loss functions
3. **Compliance requirements**: Built explainability layer for regulatory audits
4. **Integration complexity**: Seamlessly integrated with 3 legacy systems

### Code Snippets & Documentation
Available upon request with appropriate NDAs