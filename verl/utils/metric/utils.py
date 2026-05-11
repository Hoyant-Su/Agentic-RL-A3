
from typing import Any, Dict, List
import numpy as np

def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics
