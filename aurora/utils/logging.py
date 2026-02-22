"""
AURORA Structured Logging

Module-level loggers with consistent formatting.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a structured logger for an AURORA module.

    Args:
        name: Module name (e.g., 'aurora.knowledge_graph').
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Pre-configured module loggers
def fact_logger() -> logging.Logger:
    return get_logger("aurora.fact_formalization")


def graph_logger() -> logging.Logger:
    return get_logger("aurora.knowledge_graph")


def localization_logger() -> logging.Logger:
    return get_logger("aurora.parameter_localization")


def optimizer_logger() -> logging.Logger:
    return get_logger("aurora.cascade_optimizer")


def adversarial_logger() -> logging.Logger:
    return get_logger("aurora.adversarial_suite")


def certification_logger() -> logging.Logger:
    return get_logger("aurora.certification")


def quantum_logger() -> logging.Logger:
    return get_logger("aurora.quantum_distinguishability")


def pipeline_logger() -> logging.Logger:
    return get_logger("aurora.pipeline")
