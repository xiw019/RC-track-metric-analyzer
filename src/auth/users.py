"""
Simple user management with JSON-based storage.
Uses PBKDF2-HMAC for password hashing with backward compatibility
for legacy SHA-256 hashes.
"""

import os
import json
import hashlib
import secrets
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User data model."""
    username: str
    password_hash: str
    salt: str
    created_at: str
    last_login: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(**data)


class UserManager:
    """
    Manages user registration, login, and persistence.
    Uses JSON file storage with PBKDF2-HMAC password hashing.
    Backward-compatible with legacy SHA-256 hashes (auto-upgrades on login).
    """
    
    PBKDF2_ITERATIONS = 600_000
    
    def __init__(self, users_file: str = None):
        """
        Initialize user manager.
        
        Args:
            users_file: Path to JSON file for user storage.
                       Defaults to data/users.json in project root.
        """
        if users_file is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            users_file = os.path.join(data_dir, "users.json")
        
        self.users_file = users_file
        self._users: Dict[str, User] = {}
        self._load_users()
    
    def _load_users(self) -> None:
        """Load users from JSON file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, "r") as f:
                    data = json.load(f)
                    self._users = {
                        username: User.from_dict(user_data)
                        for username, user_data in data.items()
                    }
                logger.info("Loaded %d users from %s", len(self._users), self.users_file)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load users file %s: %s", self.users_file, e)
                self._users = {}
        else:
            self._users = {}
    
    def _save_users(self) -> None:
        """Save users to JSON file (atomic write via temp file + rename)."""
        data = {username: user.to_dict() for username, user in self._users.items()}
        tmp_path = self.users_file + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.users_file)
    
    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """Hash password with PBKDF2-HMAC-SHA256 (600k iterations)."""
        dk = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            iterations=UserManager.PBKDF2_ITERATIONS
        )
        return f"pbkdf2:{UserManager.PBKDF2_ITERATIONS}:{dk.hex()}"
    
    @staticmethod
    def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
        """
        Verify password against stored hash.
        Supports both new PBKDF2 format and legacy SHA-256 format.
        """
        if stored_hash.startswith("pbkdf2:"):
            # New PBKDF2 format: "pbkdf2:<iterations>:<hex_hash>"
            parts = stored_hash.split(":", 2)
            iterations = int(parts[1])
            expected_hex = parts[2]
            dk = hashlib.pbkdf2_hmac(
                'sha256', password.encode(), salt.encode(), iterations=iterations
            )
            return dk.hex() == expected_hex
        else:
            # Legacy SHA-256 format (backward compatibility)
            legacy_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return legacy_hash == stored_hash
    
    def register(self, username: str, password: str) -> tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Desired username (3-20 chars, alphanumeric + underscore)
            password: Password (min 8 chars)
        
        Returns:
            (success, message) tuple
        """
        username = username.strip().lower()
        if not username:
            return False, "Username cannot be empty"
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        if len(username) > 20:
            return False, "Username cannot exceed 20 characters"
        if not username.replace("_", "").isalnum():
            return False, "Username can only contain letters, numbers, and underscores"
        
        if username in self._users:
            return False, "Username already taken"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        
        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            created_at=datetime.now().isoformat()
        )
        
        self._users[username] = user
        self._save_users()
        logger.info("Registered new user: %s", username)
        
        return True, f"✅ User '{username}' registered successfully!"
    
    def login(self, username: str, password: str) -> tuple[bool, str, Optional[User]]:
        """
        Authenticate a user.
        
        Returns:
            (success, message, user) tuple
        """
        username = username.strip().lower()
        
        if not username or not password:
            return False, "Username and password required", None
        
        user = self._users.get(username)
        if user is None:
            return False, "Invalid username or password", None
        
        if not self._verify_password(password, user.salt, user.password_hash):
            return False, "Invalid username or password", None
        
        # Auto-upgrade legacy SHA-256 hash to PBKDF2 on successful login
        if not user.password_hash.startswith("pbkdf2:"):
            user.password_hash = self._hash_password(password, user.salt)
            logger.info("Upgraded password hash for user: %s", username)
        
        user.last_login = datetime.now().isoformat()
        self._save_users()
        logger.info("User logged in: %s", username)
        
        return True, f"✅ Welcome back, {username}!", user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self._users.get(username.strip().lower())
    
    def user_exists(self, username: str) -> bool:
        """Check if username exists."""
        return username.strip().lower() in self._users
    
    def get_user_count(self) -> int:
        """Get total number of registered users."""
        return len(self._users)
