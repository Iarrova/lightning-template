class FrameworkException(Exception):
    """Base exception for all framework-related errors."""

    pass


class ConfigurationError(FrameworkException):
    """Exception raised for errors in the configuration."""

    pass


class DatasetError(FrameworkException):
    """Exception raised for errors related to datasets."""

    pass


class ModelError(FrameworkException):
    """Exception raised for errors related to models."""

    pass


class NetworkError(FrameworkException):
    """Exception raised for errors related to networks."""

    pass


class TrainingError(FrameworkException):
    """Exception raised for errors during training."""

    pass


class ResourceNotFoundError(FrameworkException):
    """Exception raised when a required resource is not found."""

    pass
