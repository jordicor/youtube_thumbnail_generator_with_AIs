"""
YouTube Thumbnail Generator - Face Clustering Module
=====================================================
Groups detected faces into clusters using DBSCAN over facial embeddings.
Each cluster represents a different person or "character" in the video.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import orjson
import shutil

from utils import setup_logger
from config import EXPRESSION_DISTRIBUTION

logger = setup_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# DBSCAN parameters
CLUSTERING_EPS = 0.5           # Max distance between samples in same cluster
CLUSTERING_MIN_SAMPLES = 3     # Min samples to form a cluster
CLUSTERING_METRIC = 'cosine'   # Distance metric (cosine for facial embeddings)

# Cluster filtering
MIN_CLUSTER_FRAMES = 5         # Minimum frames to consider a valid cluster
MAX_CLUSTERS = 10              # Maximum clusters to return


# =============================================================================
# MAIN CLUSTERING FUNCTIONS
# =============================================================================

def cluster_faces(
    faces_data: List[Dict[str, Any]],
    eps: float = CLUSTERING_EPS,
    min_samples: int = CLUSTERING_MIN_SAMPLES
) -> Dict[str, Any]:
    """
    Group faces into clusters based on their embeddings.

    Args:
        faces_data: List of dicts with face info. Each dict must have:
                   - 'embedding': list[float] (512D vector)
                   - 'frame_path': str
                   - 'quality_score': float (optional)
                   - 'expression': str (optional)
        eps: Epsilon for DBSCAN (max distance between cluster points)
        min_samples: Minimum samples to form a cluster

    Returns:
        Dict with:
        - 'clusters': List of clusters with frames and representative
        - 'outliers': Frames not belonging to any cluster
        - 'num_clusters': Number of clusters found
        - 'total_faces': Total faces processed
    """
    if not faces_data:
        logger.warning("No faces data provided for clustering")
        return {'clusters': [], 'outliers': [], 'num_clusters': 0, 'total_faces': 0}

    # Filter faces with valid embeddings (512D)
    valid_faces = [
        f for f in faces_data
        if f.get('embedding') and len(f['embedding']) == 512
    ]

    if len(valid_faces) < min_samples:
        logger.warning(f"Not enough valid faces for clustering: {len(valid_faces)} < {min_samples}")
        return {
            'clusters': [],
            'outliers': valid_faces,
            'num_clusters': 0,
            'total_faces': len(valid_faces)
        }

    logger.info(f"Clustering {len(valid_faces)} faces with DBSCAN (eps={eps}, min_samples={min_samples})")

    # Extract embeddings as numpy array
    embeddings = np.array([f['embedding'] for f in valid_faces])

    # Normalize embeddings (important for cosine distance)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms

    # Compute cosine distance matrix
    distance_matrix = cosine_distances(embeddings_normalized)

    # DBSCAN clustering with precomputed distances
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )
    labels = clustering.fit_predict(distance_matrix)

    # Organize results by cluster
    clusters_dict = {}
    outliers = []

    for idx, label in enumerate(labels):
        face_info = valid_faces[idx].copy()

        if label == -1:
            # Outlier: doesn't belong to any cluster
            outliers.append(face_info)
        else:
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(face_info)

    # Process and filter clusters
    processed_clusters = []

    for cluster_id, faces in sorted(clusters_dict.items()):
        if len(faces) < MIN_CLUSTER_FRAMES:
            # Too few frames, treat as outliers
            outliers.extend(faces)
            continue

        # Calculate cluster centroid
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find most representative frame (closest to centroid)
        distances_to_centroid = cosine_distances([centroid], cluster_embeddings)[0]
        representative_idx = np.argmin(distances_to_centroid)
        representative_frame = faces[representative_idx]

        # Sort frames by quality score (descending)
        faces_sorted = sorted(
            faces,
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )

        processed_clusters.append({
            'cluster_index': len(processed_clusters),
            'num_frames': len(faces),
            'representative_frame': representative_frame.get('frame_path', ''),
            'representative_quality': representative_frame.get('quality_score', 0),
            'centroid': centroid.tolist(),
            'frames': faces_sorted,
            'expression_distribution': _count_expressions(faces)
        })

    # Sort clusters by number of frames (descending)
    processed_clusters.sort(key=lambda x: x['num_frames'], reverse=True)

    # Re-index after sorting
    for i, cluster in enumerate(processed_clusters):
        cluster['cluster_index'] = i

    # Limit number of clusters
    if len(processed_clusters) > MAX_CLUSTERS:
        logger.warning(f"Too many clusters ({len(processed_clusters)}), keeping top {MAX_CLUSTERS}")
        extra_clusters = processed_clusters[MAX_CLUSTERS:]
        for cluster in extra_clusters:
            outliers.extend(cluster['frames'])
        processed_clusters = processed_clusters[:MAX_CLUSTERS]

    logger.info(f"Found {len(processed_clusters)} clusters, {len(outliers)} outliers")

    return {
        'clusters': processed_clusters,
        'outliers': outliers,
        'num_clusters': len(processed_clusters),
        'total_faces': len(valid_faces)
    }


def _count_expressions(faces: List[Dict]) -> Dict[str, int]:
    """Count expression distribution in a cluster."""
    counts = {}
    for face in faces:
        expr = face.get('expression', 'unknown')
        counts[expr] = counts.get(expr, 0) + 1
    return counts


# =============================================================================
# FRAME SELECTION
# =============================================================================

def select_best_frames_from_cluster(
    cluster: Dict[str, Any],
    num_frames: int = 10,
    preferred_expression: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Select the best frames from a cluster for use as reference.

    Args:
        cluster: Cluster dict with 'frames' key
        num_frames: Number of frames to select
        preferred_expression: Preferred expression ('smiling', 'mouth_closed', etc.)

    Returns:
        List of best frames
    """
    frames = cluster.get('frames', [])

    if not frames:
        return []

    if len(frames) <= num_frames:
        return frames

    # If preferred expression, prioritize those frames
    if preferred_expression:
        preferred = [f for f in frames if f.get('expression') == preferred_expression]
        others = [f for f in frames if f.get('expression') != preferred_expression]

        # Sort each group by quality
        preferred.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        others.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        # Take 70% from preferred, fill with others
        num_preferred = min(len(preferred), int(num_frames * 0.7))
        num_others = num_frames - num_preferred

        selected = preferred[:num_preferred] + others[:num_others]
    else:
        # Without preference: diversify expressions
        selected = _select_diverse_expressions(frames, num_frames)

    return selected[:num_frames]


def _select_diverse_expressions(frames: List[Dict], num_frames: int) -> List[Dict]:
    """Select frames with expression diversity."""
    expression_groups = {}

    for frame in frames:
        expr = frame.get('expression', 'unknown')
        if expr not in expression_groups:
            expression_groups[expr] = []
        expression_groups[expr].append(frame)

    # Sort each group by quality
    for expr in expression_groups:
        expression_groups[expr].sort(
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )

    # Use configurable distribution (excluding "random" which is handled separately)
    distribution = {k: v for k, v in EXPRESSION_DISTRIBUTION.items() if k != "random"}

    selected = []

    # Select according to distribution
    for expr, count in distribution.items():
        if expr in expression_groups:
            selected.extend(expression_groups[expr][:count])

    # Fill with remaining high-quality frames (equivalent to "random" category)
    all_remaining = []
    for expr, group in expression_groups.items():
        start_idx = distribution.get(expr, 0)
        all_remaining.extend(group[start_idx:])

    all_remaining.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

    while len(selected) < num_frames and all_remaining:
        selected.append(all_remaining.pop(0))

    return selected


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def save_cluster_representatives(
    clusters: List[Dict[str, Any]],
    output_dir: Path,
    copy_frames: bool = True
) -> Path:
    """
    Save representative frames from each cluster to a directory.

    Structure (per cluster):
    - clusters/cluster_X/preview/  - Best frames for UI grid (5 frames)

    NOTE: Frame data is stored as virtual references in the database.
    The actual frames remain in output/{video}/frames/ directory.
    This saves disk space and simplifies frame management.

    Args:
        clusters: List of clusters (each with 'frames' containing all faces)
        output_dir: Base output directory
        copy_frames: If True, copy preview frames; if False, only save paths

    Returns:
        Path to clusters directory
    """
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        cluster_idx = cluster['cluster_index']
        cluster_subdir = clusters_dir / f"cluster_{cluster_idx}"
        cluster_subdir.mkdir(exist_ok=True)

        # Create preview directory only (frames are virtual references in DB)
        preview_dir = cluster_subdir / "preview"
        preview_dir.mkdir(exist_ok=True)

        # Move any existing files in cluster root to preview/ (migration of old structure)
        for old_file in list(cluster_subdir.glob("*.jpg")):
            if old_file.is_file():
                shutil.move(str(old_file), str(preview_dir / old_file.name))

        # Copy representative frame to preview/
        rep_frame = cluster.get('representative_frame', '')
        if copy_frames and rep_frame and Path(rep_frame).exists():
            dest = preview_dir / "representative.jpg"
            shutil.copy2(rep_frame, dest)
            cluster['representative_frame_local'] = str(dest)

        # Copy top 4 frames to preview/ (+ representative = 5 total for preview)
        for i, frame in enumerate(cluster.get('frames', [])[:4]):
            frame_path = frame.get('frame_path', '')
            if copy_frames and frame_path and Path(frame_path).exists():
                dest = preview_dir / f"frame_{i}.jpg"
                shutil.copy2(frame_path, dest)

        # NOTE: We no longer copy ALL frames to a frames/ subdirectory.
        # Frame paths are stored in the database as virtual references
        # pointing to the original frames in output/{video}/frames/
        logger.info(f"Cluster {cluster_idx}: {cluster['num_frames']} frames (virtual references)")

    # Save metadata
    metadata = {
        'num_clusters': len(clusters),
        'clusters': [
            {
                'index': c['cluster_index'],
                'num_frames': c['num_frames'],
                'representative': c.get('representative_frame_local', c.get('representative_frame', '')),
                'expressions': c['expression_distribution']
            }
            for c in clusters
        ]
    }

    metadata_path = clusters_dir / "clusters.json"
    with open(metadata_path, 'wb') as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

    logger.info(f"Saved {len(clusters)} cluster representatives to {clusters_dir}")

    return clusters_dir


# =============================================================================
# INTEGRATION WITH PIPELINE
# =============================================================================

def load_faces_with_embeddings(faces_json_path: Path) -> List[Dict[str, Any]]:
    """
    Load faces.json that includes embeddings.

    Args:
        faces_json_path: Path to faces.json

    Returns:
        List of faces with embeddings
    """
    if not faces_json_path.exists():
        logger.error(f"Faces JSON not found: {faces_json_path}")
        return []

    with open(faces_json_path, 'rb') as f:
        data = orjson.loads(f.read())

    # Handle both list format and dict with 'all_faces' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'all_faces' in data:
        return data['all_faces']
    else:
        logger.error("Invalid faces.json format")
        return []


def run_clustering_pipeline(
    video_output_dir: Path,
    eps: float = CLUSTERING_EPS,
    min_samples: int = CLUSTERING_MIN_SAMPLES,
    save_representatives: bool = True
) -> Dict[str, Any]:
    """
    Run the complete clustering pipeline for an analyzed video.

    Args:
        video_output_dir: Directory output/{video}/ with faces.json
        eps: Epsilon for DBSCAN
        min_samples: Minimum samples for cluster
        save_representatives: Whether to copy representative frames

    Returns:
        Clustering result
    """
    faces_json = video_output_dir / "faces.json"

    # Load faces data
    faces_data = load_faces_with_embeddings(faces_json)

    if not faces_data:
        return {'error': 'No faces data found', 'clusters': [], 'num_clusters': 0}

    # Check if embeddings are available
    faces_with_embeddings = [f for f in faces_data if f.get('embedding')]
    if not faces_with_embeddings:
        logger.warning("No embeddings found in faces.json - faces were extracted without embeddings")
        return {
            'error': 'No embeddings in faces.json - re-extract faces to enable clustering',
            'clusters': [],
            'num_clusters': 0
        }

    # Run clustering
    result = cluster_faces(faces_data, eps=eps, min_samples=min_samples)

    # Save representatives if clusters found
    if result['clusters'] and save_representatives:
        save_cluster_representatives(result['clusters'], video_output_dir)

    # Save clustering result (without full embeddings to save space)
    clusters_result_path = video_output_dir / "clustering_result.json"

    result_to_save = {
        'num_clusters': result['num_clusters'],
        'total_faces': result['total_faces'],
        'num_outliers': len(result['outliers']),
        'parameters': {
            'eps': eps,
            'min_samples': min_samples
        },
        'clusters': [
            {
                'cluster_index': c['cluster_index'],
                'num_frames': c['num_frames'],
                'representative_frame': c['representative_frame'],
                'representative_quality': c.get('representative_quality', 0),
                'expression_distribution': c['expression_distribution'],
                'top_frames': [f.get('frame_path', '') for f in c['frames'][:10]]
            }
            for c in result['clusters']
        ]
    }

    with open(clusters_result_path, 'wb') as f:
        f.write(orjson.dumps(result_to_save, option=orjson.OPT_INDENT_2))

    logger.info(f"Clustering result saved to {clusters_result_path}")

    return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_dir = Path(sys.argv[1])
    else:
        video_dir = Path("output/test_video")

    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        print("Usage: python face_clustering.py <output_video_directory>")
        sys.exit(1)

    result = run_clustering_pipeline(video_dir)

    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        print(f"\nClustering complete:")
        print(f"  - Total faces: {result['total_faces']}")
        print(f"  - Clusters found: {result['num_clusters']}")
        print(f"  - Outliers: {len(result['outliers'])}")

        for cluster in result['clusters']:
            print(f"\n  Cluster {cluster['cluster_index']}:")
            print(f"    - Frames: {cluster['num_frames']}")
            print(f"    - Expressions: {cluster['expression_distribution']}")
            print(f"    - Representative: {cluster['representative_frame']}")
