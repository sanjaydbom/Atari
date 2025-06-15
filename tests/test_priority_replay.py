import pytest
import numpy as np
from unittest.mock import patch
from src.priority_replay import PriorityReplay  # Replace with your actual module name

class TestSumTree:
    @pytest.fixture
    def sum_tree(self):
        return PriorityReplay(capacity=4)

    def test_init(self, sum_tree):
        """Test initialization of SumTree."""
        assert sum_tree.capacity == 4
        assert len(sum_tree.tree) == 8  # 2 * capacity for complete binary tree
        assert len(sum_tree.data) == 4  # Data array matches capacity
        assert sum_tree.size == 0
        assert sum_tree.data_pointer == 0
        assert sum_tree.get_total() == 0

    def test_add_single(self, sum_tree):
        """Test adding a single element."""
        sum_tree.add(priority=2.0, element="data1")
        assert sum_tree.size == 1
        assert sum_tree.data[0] == "data1"
        assert sum_tree.tree[sum_tree.capacity - 1] == 2.0  # Leaf node
        assert sum_tree.get_total() == 2.0  # Root node
        assert sum_tree.tree[0] == 2.0  # Root node

    def test_add_multiple(self, sum_tree):
        """Test adding multiple elements."""
        sum_tree.add(priority=2.0, element="data1")
        sum_tree.add(priority=3.0, element="data2")
        assert sum_tree.size == 2
        assert sum_tree.data[0] == "data1"
        assert sum_tree.data[1] == "data2"
        assert sum_tree.tree[sum_tree.capacity - 1] == 2.0
        assert sum_tree.tree[sum_tree.capacity] == 3.0
        assert sum_tree.get_total() == 5.0
        assert sum_tree.tree[0] == 5.0

    def test_add_beyond_capacity(self, sum_tree):
        """Test adding elements beyond capacity (should overwrite)."""
        for i in range(5):
            sum_tree.add(priority=float(i + 1), element=f"data{i + 1}")
        assert sum_tree.size == 4  # Size capped at capacity
        assert sum_tree.data[0] == "data5"  # Overwritten
        assert sum_tree.data[1] == "data2"
        assert sum_tree.data[2] == "data3"
        assert sum_tree.data[3] == "data4"
        assert sum_tree.get_total() == 14.0  # 5 + 2 + 3 + 4

    def test_get_total_empty(self, sum_tree):
        """Test get_total on an empty tree."""
        assert sum_tree.get_total() == 0

    def test_sample_empty(self, sum_tree):
        """Test sampling from an empty tree."""  # Adjust based on your implementation
        assert sum_tree.sample() == None

    @patch('numpy.random.uniform')
    def test_sample_single(self, mock_random, sum_tree):
        """Test sampling with a single element."""
        sum_tree.add(priority=2.0, element="data1")
        mock_random.return_value = 1.0
        idx, priority, data = sum_tree.sample()
        assert idx == sum_tree.capacity - 1
        assert priority == 2.0
        assert data == "data1"

    @patch('random.uniform')
    def test_sample_multiple(self, mock_random, sum_tree):
        """Test sampling with multiple elements."""
        # Add elements
        sum_tree.add(priority=2.0, element="data1")
        sum_tree.add(priority=3.0, element="data2")
        sum_tree.add(priority=1.0, element="data3")

        # Verify tree state
        assert sum_tree.size == 3, f"Expected size 3, got {sum_tree.size}"
        assert sum_tree.get_total() == 6.0, f"Expected total sum 6.0, got {sum_tree.get_total()}"
        assert sum_tree.tree[sum_tree.capacity - 1] == 2.0, f"Expected leaf[0] = 2.0, got {sum_tree.tree[sum_tree.capacity - 1]}"
        assert sum_tree.tree[sum_tree.capacity] == 3.0, f"Expected leaf[1] = 3.0, got {sum_tree.tree[sum_tree.capacity]}"
        assert sum_tree.tree[sum_tree.capacity + 1] == 1.0, f"Expected leaf[2] = 1.0, got {sum_tree.tree[sum_tree.capacity + 1]}"

        # Mock random values to hit specific leaves
        mock_random.side_effect = [0.5, 2.5, 5.5]  # Changed 4.5 to 5.5 for data3

        # Sample first element (s = 0.5, should pick data1)
        idx, priority, data = sum_tree.sample()
        assert idx == sum_tree.capacity - 1, f"Expected idx {sum_tree.capacity - 1}, got {idx}"
        assert priority == 2.0, f"Expected priority 2.0, got {priority}"
        assert data == "data1", f"Expected data 'data1', got {data}"

        # Sample second element (s = 2.5, should pick data2)
        idx, priority, data = sum_tree.sample()
        assert idx == sum_tree.capacity, f"Expected idx {sum_tree.capacity}, got {idx}"
        assert priority == 3.0, f"Expected priority 3.0, got {priority}"
        assert data == "data2", f"Expected data 'data2', got {data}"

        # Sample third element (s = 5.5, should pick data3)
        idx, priority, data = sum_tree.sample()
        assert idx == sum_tree.capacity + 1, f"Expected idx {sum_tree.capacity + 1}, got {idx}"
        assert priority == 1.0, f"Expected priority 1.0, got {priority}"
        assert data == "data3", f"Expected data 'data3', got {data}"

    def test_sample_distribution(self, sum_tree):
        """Test sampling distribution (statistical test)."""
        sum_tree.add(priority=1.0, element="data1")
        sum_tree.add(priority=3.0, element="data2")
        sum_tree.add(priority=1.0, element="data3")

        # Run multiple samples and check distribution
        samples = {0: 0, 1: 0, 2: 0}
        n_samples = 10000
        for _ in range(n_samples):
            idx, _, _ = sum_tree.sample()
            samples[idx - (sum_tree.capacity - 1)] += 1

        # Expected probabilities: 1/5, 3/5, 1/5
        expected = [0.2, 0.6, 0.2]
        observed = [samples[i] / n_samples for i in range(3)]
        for o, e in zip(observed, expected):
            assert abs(o - e) < 0.05  # Allow small statistical variation

if __name__ == '__main__':
    pytest.main()