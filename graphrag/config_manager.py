#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class GraphRAGConfig:
    """GraphRAG configuration class with validation and defaults."""

    use_graphrag: bool = False
    method: str = "light"  # "light" or "general"
    # 默认实体类型配置：更新为电池材料研发核心实体
    entity_types: List[str] = field(default_factory=lambda: ["正极基体", "掺杂剂", "包覆相", "工艺参数", "性能指标", "失效机理"])
    resolution: bool = False
    community: bool = False
    max_entities_per_chunk: int = 50
    max_relations_per_chunk: int = 100
    timeout_seconds: int = 3600  # 1 hour default timeout
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate the GraphRAG configuration."""
        if self.method not in ["light", "general"]:
            raise ValueError(f"Invalid method '{self.method}'. Must be 'light' or 'general'")
        
        if not isinstance(self.entity_types, list) or not self.entity_types:
            raise ValueError("entity_types must be a non-empty list")
        
        if self.max_entities_per_chunk <= 0:
            raise ValueError("max_entities_per_chunk must be positive")
        
        if self.max_relations_per_chunk <= 0:
            raise ValueError("max_relations_per_chunk must be positive")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "use_graphrag": self.use_graphrag,
            "method": self.method,
            "entity_types": self.entity_types,
            "resolution": self.resolution,
            "community": self.community,
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "max_relations_per_chunk": self.max_relations_per_chunk,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
        """Create configuration from dictionary with validation."""
        # Filter out unknown keys and provide defaults
        known_keys = {
            "use_graphrag", "method", "entity_types", "resolution", 
            "community", "max_entities_per_chunk", "max_relations_per_chunk", 
            "timeout_seconds"
        }
        
        filtered_config = {k: v for k, v in config_dict.items() if k in known_keys}
        return cls(**filtered_config)


class GraphRAGConfigManager:
    """Manager for GraphRAG configuration validation and processing."""

    # 电池材料研发核心实体类型默认列表
    DEFAULT_ENTITY_TYPES = ["正极基体", "掺杂剂", "包覆相", "工艺参数", "性能指标", "失效机理"]
    SUPPORTED_METHODS = ["light", "general"]
    
    @staticmethod
    def validate_task_config(task: Dict[str, Any]) -> GraphRAGConfig:
        """
        Validate and normalize GraphRAG configuration from a task.
        
        Args:
            task: Task dictionary containing configuration
            
        Returns:
            GraphRAGConfig: Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Extract GraphRAG configuration
            parser_config = task.get("kb_parser_config", {})
            graphrag_config = parser_config.get("graphrag", {})
            
            if not graphrag_config:
                # Return default disabled configuration
                return GraphRAGConfig(use_graphrag=False)
            
            # Create and validate configuration
            config = GraphRAGConfig.from_dict(graphrag_config)
            
            # Additional task-specific validation
            if config.use_graphrag:
                tenant_id = task.get("tenant_id")
                kb_id = task.get("kb_id")
                doc_id = task.get("doc_id")
                
                if not all([tenant_id, kb_id, doc_id]):
                    raise ValueError(f"Missing required task parameters: tenant_id={tenant_id}, kb_id={kb_id}, doc_id={doc_id}")
            
            return config
            
        except Exception as e:
            logging.error(f"GraphRAG configuration validation failed: {e}")
            raise ValueError(f"Invalid GraphRAG configuration: {str(e)}")
    
    @staticmethod
    def get_default_config() -> GraphRAGConfig:
        """Get default GraphRAG configuration."""
        return GraphRAGConfig()
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries with override precedence.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = GraphRAGConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def estimate_processing_time(config: GraphRAGConfig, chunk_count: int) -> int:
        """
        Estimate processing time in seconds based on configuration and chunk count.
        
        Args:
            config: GraphRAG configuration
            chunk_count: Number of chunks to process
            
        Returns:
            int: Estimated processing time in seconds
        """
        if not config.use_graphrag:
            return 0
        
        # Base time per chunk (in seconds)
        base_time_per_chunk = 30 if config.method == "light" else 60
        
        # Additional time for resolution and community detection
        resolution_multiplier = 1.5 if config.resolution else 1.0
        community_multiplier = 1.3 if config.community else 1.0
        
        estimated_time = int(
            chunk_count * base_time_per_chunk * resolution_multiplier * community_multiplier
        )
        
        # Add minimum and maximum bounds
        min_time = 60  # 1 minute minimum
        max_time = config.timeout_seconds
        
        return max(min_time, min(estimated_time, max_time))
    
    @staticmethod
    def validate_entity_types(entity_types: List[str]) -> List[str]:
        """
        Validate and normalize entity types.
        
        Args:
            entity_types: List of entity types to validate
            
        Returns:
            List[str]: Validated and normalized entity types
        """
        if not entity_types or not isinstance(entity_types, list):
            return GraphRAGConfigManager.DEFAULT_ENTITY_TYPES.copy()
        
        # Normalize entity types (lowercase, strip whitespace)
        normalized = []
        for entity_type in entity_types:
            if isinstance(entity_type, str) and entity_type.strip():
                normalized.append(entity_type.strip().lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_types = []
        for entity_type in normalized:
            if entity_type not in seen:
                seen.add(entity_type)
                unique_types.append(entity_type)
        
        # Return default if no valid types found
        return unique_types if unique_types else GraphRAGConfigManager.DEFAULT_ENTITY_TYPES.copy()


def create_graphrag_task(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    config: GraphRAGConfig,
    language: str = "Chinese",
    llm_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a GraphRAG task dictionary with proper configuration.
    
    Args:
        tenant_id: Tenant identifier
        kb_id: Knowledge base identifier
        doc_id: Document identifier
        config: GraphRAG configuration
        language: Processing language
        llm_id: LLM model identifier
        
    Returns:
        Dict[str, Any]: Task dictionary ready for execution
    """
    task = {
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "task_type": "graphrag",
        "language": language,
        "kb_parser_config": {
            "graphrag": config.to_dict()
        }
    }
    
    if llm_id:
        task["llm_id"] = llm_id
    
    return task
