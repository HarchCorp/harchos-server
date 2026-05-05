"""Enumerations for HarchOS — strict validation for all enum-like fields.

These replace the raw string fields that previously accepted any value.
Every enum member maps to the string values already stored in the database
to maintain backward compatibility.
"""

from enum import Enum


class WorkloadType(str, Enum):
    """Type of GPU workload."""
    TRAINING = "training"
    INFERENCE = "inference"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    DATA_PIPELINE = "data_pipeline"
    BATCH = "batch"


class WorkloadStatus(str, Enum):
    """Lifecycle status of a workload."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkloadPriority(str, Enum):
    """Priority level for workload scheduling."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class HubStatus(str, Enum):
    """Operational status of a GPU hub."""
    CREATING = "creating"
    READY = "ready"
    UPDATING = "updating"
    SCALING = "scaling"
    DRAINING = "draining"
    OFFLINE = "offline"
    ERROR = "error"


class HubTier(str, Enum):
    """Service tier for a GPU hub."""
    STARTER = "starter"
    STANDARD = "standard"
    PERFORMANCE = "performance"
    ENTERPRISE = "enterprise"


class SovereigntyLevel(str, Enum):
    """Data sovereignty strictness level."""
    STRICT = "strict"
    STANDARD = "standard"
    MODERATE = "moderate"
    MINIMAL = "minimal"


class CarbonAction(str, Enum):
    """Carbon optimization action type."""
    SCHEDULE_NOW = "schedule_now"
    DEFER = "defer"
    REJECT = "reject"
    NO_SUITABLE_HUB = "no_suitable_hub"


class ModelFramework(str, Enum):
    """ML model framework."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    VLLM = "vllm"
    TRITON = "triton"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Operational status of a deployed model."""
    DEPLOYING = "deploying"
    READY = "ready"
    SCALING = "scaling"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class BillingStatus(str, Enum):
    """Billing record status."""
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class UserRole(str, Enum):
    """User role for RBAC."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class DataClassification(str, Enum):
    """Data classification level for sovereignty."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class InferenceModelFamily(str, Enum):
    """Model family for inference catalog."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GEMMA = "gemma"
    PHI = "phi"
    STARCODER = "starcoder"
    COHERE = "cohere"
    YI = "yi"
    SOLAR = "solar"
    EMBEDDING = "embedding"


class CarbonPreference(str, Enum):
    """Carbon-aware routing preference for inference."""
    LOWEST = "lowest"       # Route to greenest hub regardless of latency
    FASTEST = "fastest"     # Route to lowest-latency hub
    BALANCED = "balanced"   # Balance latency and carbon intensity


class Region(str, Enum):
    """Deployment region identifiers."""
    AFRICA_NORTH = "africa-north"
    AFRICA_WEST = "africa-west"
    AFRICA_EAST = "africa-east"
    AFRICA_SOUTH = "africa-south"
    EUROPE_WEST = "europe-west"
    EUROPE_NORTH = "europe-north"
    MIDDLE_EAST = "middle-east"
