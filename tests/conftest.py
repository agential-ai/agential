"""Configs for pytest."""
from tests.fixtures.agent import *
from tests.fixtures.data import *
from tests.fixtures.retriever import *
from tests.fixtures.api import *

from dotenv import load_dotenv

load_dotenv()