"""
Implementation of the Proof-of-Evolution (PoE) consensus mechanism for EternaLattice.
This consensus mechanism mimics natural selection by evolving traits over time.
"""
import random
import logging
import math
from typing import Dict, List, Tuple
import config

logger = logging.getLogger(__name__)

# Define the set of consensus traits that can evolve
CONSENSUS_TRAITS = [
    "block_time_preference",
    "storage_redundancy",
    "verification_strictness",
    "energy_efficiency",
    "connection_density",
    "data_compression",
    "encryption_strength",
    "fork_resolution_strategy"
]

def initial_traits() -> Dict[str, float]:
    """
    Generate initial consensus traits for a new node or the genesis block.
    
    Returns:
        Dict[str, float]: Dictionary of traits with initial values
    """
    traits = {}
    
    # Initialize with default values
    traits["block_time_preference"] = 0.5  # 0=fast, 1=slow
    traits["storage_redundancy"] = 0.5  # 0=minimal, 1=maximal
    traits["verification_strictness"] = 0.5  # 0=permissive, 1=strict
    traits["energy_efficiency"] = 0.5  # 0=performance, 1=efficiency
    traits["connection_density"] = 0.5  # 0=few connections, 1=many connections
    traits["data_compression"] = 0.5  # 0=minimal, 1=maximal
    traits["encryption_strength"] = 0.5  # 0=speed, 1=security
    traits["fork_resolution_strategy"] = 0.5  # 0=first-seen, 1=most-work
    
    return traits

def mutate_trait(value: float, mutation_rate: float) -> float:
    """
    Mutate a trait value randomly based on the mutation rate.
    
    Args:
        value: Current trait value
        mutation_rate: Probability of mutation
        
    Returns:
        float: Mutated trait value
    """
    if random.random() < mutation_rate:
        # Apply mutation
        change = random.gauss(0, 0.1)  # Normal distribution with mean 0, std 0.1
        new_value = value + change
        
        # Ensure value stays within [0, 1]
        return max(0.0, min(1.0, new_value))
    
    return value

def crossover(parent1: Dict[str, float], parent2: Dict[str, float], crossover_rate: float) -> Dict[str, float]:
    """
    Perform crossover between two parents to produce a child.
    
    Args:
        parent1: First parent traits
        parent2: Second parent traits
        crossover_rate: Probability of crossover
        
    Returns:
        Dict[str, float]: Child traits
    """
    if random.random() < crossover_rate:
        # Perform crossover
        child = {}
        
        for trait in CONSENSUS_TRAITS:
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child[trait] = parent1[trait]
            else:
                child[trait] = parent2[trait]
    else:
        # No crossover, just copy parent1
        child = parent1.copy()
    
    return child

def calculate_fitness(traits: Dict[str, float]) -> float:
    """
    Calculate fitness score for a set of traits.
    The fitness function evaluates how well the traits work together.
    
    Args:
        traits: Dictionary of trait values
        
    Returns:
        float: Fitness score
    """
    fitness = 0.0
    
    # Reward for balanced energy efficiency and performance
    energy_score = 1.0 - 2.0 * abs(traits["energy_efficiency"] - 0.5)
    fitness += energy_score
    
    # Reward for correlation between verification strictness and encryption strength
    security_correlation = 1.0 - abs(traits["verification_strictness"] - traits["encryption_strength"])
    fitness += security_correlation
    
    # Reward for appropriate storage redundancy based on connection density
    redundancy_score = 1.0 - abs(traits["storage_redundancy"] - (1.0 - traits["connection_density"]))
    fitness += redundancy_score
    
    # Reward for balance between data compression and block time
    compression_time_balance = 1.0 - abs(traits["data_compression"] - traits["block_time_preference"])
    fitness += compression_time_balance
    
    # Return normalized fitness score
    return fitness / 4.0

def select_parents(population: List[Dict[str, float]], fitness_scores: List[float], 
                 selection_pressure: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Select two parents using fitness proportional selection.
    
    Args:
        population: List of trait dictionaries
        fitness_scores: List of fitness scores
        selection_pressure: Selection pressure parameter
        
    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Two selected parents
    """
    # Apply selection pressure to fitness scores
    adjusted_scores = [score ** selection_pressure for score in fitness_scores]
    total_fitness = sum(adjusted_scores)
    
    if total_fitness == 0:
        # If all fitness scores are 0, select randomly
        return random.sample(population, 2)
    
    # Normalize probabilities
    probabilities = [score / total_fitness for score in adjusted_scores]
    
    # Select parents using fitness proportional selection
    parent1_idx = random.choices(range(len(population)), weights=probabilities)[0]
    parent1 = population[parent1_idx]
    
    # Make sure we don't select the same parent twice
    remaining_population = population.copy()
    remaining_population.pop(parent1_idx)
    remaining_fitness = fitness_scores.copy()
    remaining_fitness.pop(parent1_idx)
    
    if not remaining_population:
        # If only one individual in population, duplicate it
        return parent1, parent1
    
    # Recalculate probabilities
    adjusted_remaining = [score ** selection_pressure for score in remaining_fitness]
    total_remaining = sum(adjusted_remaining)
    
    if total_remaining == 0:
        # If all remaining fitness scores are 0, select randomly
        parent2 = random.choice(remaining_population)
    else:
        remaining_probs = [score / total_remaining for score in adjusted_remaining]
        parent2_idx = random.choices(range(len(remaining_population)), weights=remaining_probs)[0]
        parent2 = remaining_population[parent2_idx]
    
    return parent1, parent2

def evolve_traits(population: List[Dict[str, float]], fitness_scores: List[float]) -> Dict[str, float]:
    """
    Evolve a new set of consensus traits based on the current population.
    
    Args:
        population: List of trait dictionaries
        fitness_scores: List of fitness scores
        
    Returns:
        Dict[str, float]: Evolved traits
    """
    if not population:
        logger.warning("Empty population for evolution")
        return initial_traits()
    
    # Configuration parameters
    mutation_rate = config.POE_MUTATION_RATE
    crossover_rate = config.POE_CROSSOVER_RATE
    selection_pressure = config.POE_SELECTION_PRESSURE
    
    # Number of children to produce
    num_children = max(5, len(population))
    
    # Generate children
    children = []
    children_fitness = []
    
    for _ in range(num_children):
        # Select parents
        parent1, parent2 = select_parents(population, fitness_scores, selection_pressure)
        
        # Perform crossover
        child = crossover(parent1, parent2, crossover_rate)
        
        # Apply mutations
        for trait in CONSENSUS_TRAITS:
            child[trait] = mutate_trait(child[trait], mutation_rate)
        
        # Calculate fitness
        fitness = calculate_fitness(child)
        
        children.append(child)
        children_fitness.append(fitness)
    
    # Find the best child
    best_child_idx = children_fitness.index(max(children_fitness))
    best_child = children[best_child_idx]
    
    logger.info(f"Evolved traits with fitness {children_fitness[best_child_idx]}")
    
    return best_child

def get_trait_description(trait: str, value: float) -> str:
    """
    Get a human-readable description of a trait value.
    
    Args:
        trait: Trait name
        value: Trait value
        
    Returns:
        str: Description of the trait value
    """
    descriptions = {
        "block_time_preference": {
            "low": "Prioritizes fast block times",
            "mid": "Balanced block time preference",
            "high": "Prioritizes longer, more stable block times"
        },
        "storage_redundancy": {
            "low": "Minimal data redundancy",
            "mid": "Moderate data redundancy",
            "high": "Maximum data redundancy"
        },
        "verification_strictness": {
            "low": "Permissive transaction verification",
            "mid": "Standard verification requirements",
            "high": "Strict verification requirements"
        },
        "energy_efficiency": {
            "low": "Prioritizes performance over efficiency",
            "mid": "Balanced energy usage",
            "high": "Highly energy-efficient"
        },
        "connection_density": {
            "low": "Sparse node connections",
            "mid": "Moderate connection density",
            "high": "Dense node interconnections"
        },
        "data_compression": {
            "low": "Minimal data compression",
            "mid": "Standard compression",
            "high": "Maximum data compression"
        },
        "encryption_strength": {
            "low": "Fast but less secure encryption",
            "mid": "Balanced encryption",
            "high": "Maximum security encryption"
        },
        "fork_resolution_strategy": {
            "low": "First-seen fork resolution",
            "mid": "Balanced fork resolution strategy",
            "high": "Most-work fork resolution"
        }
    }
    
    # Determine category based on value
    if value < 0.33:
        category = "low"
    elif value < 0.67:
        category = "mid"
    else:
        category = "high"
    
    return descriptions[trait][category]

def get_consensus_state() -> Dict[str, str]:
    """
    Get the current state of the consensus mechanism in a human-readable format.
    
    Returns:
        Dict[str, str]: Dictionary of traits with descriptions
    """
    # In a real implementation, this would get the current network-wide consensus
    # For the prototype, we'll use a predefined set of traits
    traits = initial_traits()
    
    # Create a dictionary of descriptions
    descriptions = {}
    for trait, value in traits.items():
        descriptions[trait] = get_trait_description(trait, value)
    
    return descriptions
