"""
Firebase Client for ARLTER
Handles all Firestore operations with robust error handling and connection management
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config import FIREBASE_CONFIG, SYSTEM_CONFIG


class FirebaseClient:
    """Firebase client singleton with connection pooling and error recovery"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._db = None
            self._initialize_firebase()
            self._initialized = True
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection with error handling"""
        try:
            # Check if Firebase app is already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_CONFIG.credential_path