"""Tests for Drain template miner."""

import pytest
from app.parsers.drain_miner import DrainMiner, compute_template_hash, mine_template, reset_miners


class TestDrainMiner:
    """Test DrainMiner class."""

    @pytest.fixture
    def miner(self):
        """Create a DrainMiner instance for testing."""
        return DrainMiner()

    def test_basic_template_extraction(self, miner):
        """Test basic template extraction."""
        template_text, params, cluster_id = miner.add_log_message("User alice logged in from 192.168.1.100")
        
        assert template_text is not None
        assert cluster_id >= 0
        assert len(template_text) > 0

    def test_consistent_hashing(self, miner):
        """Test that same template produces same hash."""
        template1, _, _ = miner.add_log_message("User alice logged in from 192.168.1.100")
        template2, _, _ = miner.add_log_message("User bob logged in from 10.0.0.1")
        
        hash1 = compute_template_hash("test", template1)
        hash2 = compute_template_hash("test", template2)
        
        assert hash1 == hash2
        assert template1 == template2

    def test_different_templates_different_hashes(self, miner):
        """Test that different templates produce different hashes."""
        template1, _, _ = miner.add_log_message("User alice logged in")
        template2, _, _ = miner.add_log_message("Error connecting to database")
        
        hash1 = compute_template_hash("test", template1)
        hash2 = compute_template_hash("test", template2)
        
        assert hash1 != hash2

    def test_parameter_extraction(self, miner):
        """Test that parameters are correctly extracted."""
        template, params, _ = miner.add_log_message("Connection from 192.168.1.100 port 22")
        
        assert template is not None
        assert isinstance(params, list)

    def test_empty_input(self, miner):
        """Test handling of empty input."""
        template, params, cluster_id = miner.add_log_message("")
        assert template == "<*>"
        assert params == [""]

    def test_whitespace_input(self, miner):
        """Test handling of whitespace input."""
        template, params, cluster_id = miner.add_log_message("   ")
        assert template == "<*>"

    def test_cluster_count(self, miner):
        """Test retrieving cluster count."""
        miner.add_log_message("User alice logged in")
        miner.add_log_message("User bob logged in")
        miner.add_log_message("Error connecting to database")
        
        count = miner.cluster_count
        assert count >= 2

    def test_long_log_line(self, miner):
        """Test handling of long log lines."""
        long_log = "A" * 1000 + " with value 12345 " + "B" * 1000
        template, params, cluster_id = miner.add_log_message(long_log)
        assert template is not None

    def test_special_characters(self, miner):
        """Test handling of special characters."""
        template, params, _ = miner.add_log_message("Error: [CRITICAL] @user#123 failed with $500.00")
        assert template is not None
        hash_val = compute_template_hash("test", template)
        assert hash_val != 0


class TestComputeTemplateHash:
    """Test compute_template_hash function."""
    
    def test_hash_is_deterministic(self):
        """Test that hash is deterministic."""
        hash1 = compute_template_hash("service", "User <*> logged in")
        hash2 = compute_template_hash("service", "User <*> logged in")
        assert hash1 == hash2
    
    def test_hash_differs_by_service(self):
        """Test that hash differs by service."""
        hash1 = compute_template_hash("service1", "User <*> logged in")
        hash2 = compute_template_hash("service2", "User <*> logged in")
        assert hash1 != hash2
    
    def test_hash_is_signed_64bit(self):
        """Test that hash fits in signed 64-bit integer."""
        hash_val = compute_template_hash("service", "Some template <*>")
        assert -(2**63) <= hash_val < 2**63


class TestMineTemplate:
    """Test mine_template convenience function."""
    
    def test_mine_template(self):
        """Test mine_template function."""
        reset_miners()
        template, params, template_hash = mine_template("User alice logged in", "my-service")
        
        assert template is not None
        assert isinstance(params, list)
        assert isinstance(template_hash, int)
