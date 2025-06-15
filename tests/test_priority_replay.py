import pytest
import random
import math

from src.priority_replay import PriorityReplay


# A small capacity for easier and faster testing
TEST_CAPACITY = 8

@pytest.fixture
def replay_buffer():
    """
    Pytest fixture to create a PriorityReplay instance with a small capacity
    for each test, ensuring tests are isolated.
    """
    # This fixture assumes a dummy MEMORY_SIZE if config is not present
    try:
        from src.priority_replay import MEMORY_SIZE
    except ImportError:
        # Define a dummy value if config doesn't exist, so PriorityReplay can init
        import priority_replay
        priority_replay.MEMORY_SIZE = 100 
        
    return PriorityReplay(capacity=TEST_CAPACITY)

def test_initialization(replay_buffer):
    """
    Tests that the buffer is initialized correctly with empty values.
    """
    assert len(replay_buffer) == 0
    assert replay_buffer.size == 0
    assert replay_buffer.get_total() == 0
    assert replay_buffer.capacity == TEST_CAPACITY
    # A complete sum tree has 2*capacity - 1 nodes
    assert len(replay_buffer.tree) == 2 * TEST_CAPACITY - 1

def test_append_single_element(replay_buffer):
    """
    Tests appending a single element to the buffer and checks tree integrity.
    """
    replay_buffer.append(1.0, "experience_1")
    assert len(replay_buffer) == 1
    assert replay_buffer.get_total() == 1.0
    assert replay_buffer.data[0] == "experience_1"
    # The leaf node in the tree should have the priority
    leaf_index = replay_buffer.capacity - 1
    assert replay_buffer.tree[leaf_index] == 1.0
    # The root should also be updated
    assert replay_buffer.tree[0] == 1.0

def test_append_multiple_elements(replay_buffer):
    """
    Tests appending multiple elements and verifies the total priority (tree root).
    """
    experiences = [("e1", 0.5), ("e2", 1.0), ("e3", 0.2)]
    total_priority = 0
    for i, (exp, prio) in enumerate(experiences):
        replay_buffer.append(prio, exp)
        total_priority += prio
        assert len(replay_buffer) == i + 1
        # Use pytest.approx for floating point comparisons
        assert pytest.approx(replay_buffer.get_total()) == total_priority

def test_buffer_overwrite_logic(replay_buffer):
    """
    Tests that the buffer correctly overwrites the oldest elements when its
    capacity is exceeded.
    """
    # Fill the buffer to capacity
    for i in range(TEST_CAPACITY):
        replay_buffer.append(i + 1.0, f"experience_{i}")

    assert len(replay_buffer) == TEST_CAPACITY
    initial_total = sum(range(1, TEST_CAPACITY + 1))
    assert pytest.approx(replay_buffer.get_total()) == initial_total
    
    # Add one more element to trigger overwrite at the beginning
    replay_buffer.append(10.0, "new_experience")

    assert len(replay_buffer) == TEST_CAPACITY
    # The first element (priority 1.0) should be replaced by the new one (priority 10.0)
    expected_total = initial_total - 1.0 + 10.0
    assert pytest.approx(replay_buffer.get_total()) == expected_total
    # Check that the data was overwritten at the first position
    assert replay_buffer.data[0] == "new_experience"
    assert "experience_0" not in replay_buffer.data

def test_sample_from_empty_buffer(replay_buffer):
    """
    Tests that sampling from an empty buffer returns None as per the implementation.
    """
    result = replay_buffer.sample(4)
    assert result is None

def test_sample_returns_correct_format(replay_buffer):
    """
    Tests that the sample method returns a tuple of two lists (samples and indices)
    of the correct length.
    """
    replay_buffer.append(1.0, "e1")
    replay_buffer.append(1.0, "e2")
    
    num_samples = 2
    samples, indices = replay_buffer.sample(num_samples)

    assert isinstance(samples, list)
    assert isinstance(indices, list)
    assert len(samples) == num_samples
    assert len(indices) == num_samples
    assert "e" in samples[0] # check content
    assert isinstance(indices[0], int) # check content

def test_update_all_td(replay_buffer):
    """
    Tests the batch update functionality of update_all_td.
    """
    # Add 4 elements with initial priorities
    replay_buffer.append(1.0, "e1") # index: 7
    replay_buffer.append(2.0, "e2") # index: 8
    replay_buffer.append(3.0, "e3") # index: 9
    replay_buffer.append(4.0, "e4") # index: 10
    assert pytest.approx(replay_buffer.get_total()) == 10.0

    # The tree indices for the first 4 elements in a tree with capacity 8 are 7, 8, 9, 10
    indices_to_update = [7, 9] 
    new_priorities = [5.0, 6.0]

    replay_buffer.update_all_td(indices_to_update, new_priorities)

    # Original total was 10.0.
    # We updated index 7 from 1.0 to 5.0 (change of +4.0)
    # We updated index 9 from 3.0 to 6.0 (change of +3.0)
    # New total should be 10.0 + 4.0 + 3.0 = 17.0
    expected_total = (10.0 - 1.0 - 3.0) + 5.0 + 6.0
    assert pytest.approx(replay_buffer.get_total()) == expected_total
    assert replay_buffer.tree[7] == 5.0
    assert replay_buffer.tree[9] == 6.0

def test_priority_sampling_is_biased(replay_buffer):
    """
    Tests that elements with higher priority are sampled more frequently.
    """
    # Add one element with a very high priority and others with low priority
    replay_buffer.append(1000.0, "high_priority_exp")
    for i in range(TEST_CAPACITY - 1):
        replay_buffer.append(0.01, f"low_priority_exp_{i}")

    # Sample many times
    num_samples = 1000
    samples, _ = replay_buffer.sample(num_samples)
    
    # Count the occurrences of the high-priority sample
    high_priority_count = samples.count("high_priority_exp")

    # The high-priority element should be sampled a vast majority of the time.
    # We expect it to be chosen >95% of the time given the extreme priority difference.
    assert high_priority_count > 950

