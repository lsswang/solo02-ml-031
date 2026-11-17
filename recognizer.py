"""Working closed-set centroid recognizer with one global rejection radius."""
import numpy as np
from adapter import transform

def build(support, calibration, adapter, shrinkage=0.2):
    classes=sorted(support)
    centers={c:transform(support[c],adapter).mean(axis=0).tolist() for c in classes}
    distances=[np.linalg.norm(z-np.asarray(centers[c])) for c in classes for z in transform(calibration[c],adapter)]
    return {"classes":classes,"centers":centers,"adapter":adapter,"radius":float(max(distances))}

def predict(features,model,alpha=0.4):
    out=[]
    for z in transform(features,model["adapter"]):
        pairs=[(float(np.linalg.norm(z-np.asarray(model["centers"][c]))),c) for c in model["classes"]]
        distance,c=min(pairs)
        out.append({"decision":"unique" if distance<=model["radius"] else "unknown","classes":[c] if distance<=model["radius"] else []})
    return out
