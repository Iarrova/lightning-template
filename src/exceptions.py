class FrameworkException(Exception):
    """Base exception for all framework-related errors."""

    pass


class ConfigurationError(FrameworkException):
    """Exception raised for errors in the configuration."""

    pass
