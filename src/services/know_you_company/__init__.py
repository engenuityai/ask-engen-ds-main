"""Know Your Company service — mode="know_your_company".

Registers the KYC pipeline with src.services.registry at import time so
main.py only needs to import this package to activate the service.
"""

from src.services.know_you_company.pipeline import run_query_pipeline
from src.services.registry import register

register("know_your_company", run_query_pipeline)
