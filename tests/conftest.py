"""Configs for pytest."""

from dotenv import load_dotenv

from tests.fixtures.agent import *
from tests.fixtures.api import *
from tests.fixtures.data import *


load_dotenv()
