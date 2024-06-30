"""Inference related routes."""

import structlog
import aiohttp
import numpy as np
import pandas as pd
from fastapi import APIRouter


router = APIRouter()

log = structlog.get_logger()
