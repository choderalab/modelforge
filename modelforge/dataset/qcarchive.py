import os
from abc import ABC, abstractmethod
from typing import Tuple

import qcportal as ptl
import torch
from loguru import logger


class QCPortal():
    
    def __init__(self) -> None:
        pass
    
    def from_qcportal(self):
        # retrieval from QCPortal (data source)
        pass
    
    def to_qcportal(self):
        # upload to QCPortal (data source)
        pass
    
    