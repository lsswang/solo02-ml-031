"""Working ordinary ridge feature adapter; row-vector convention."""
import numpy as np

def fit(features, targets, anchors, ridge=0.5):
    x=np.asarray(features,dtype=float); y=np.asarray(targets,dtype=float)
    d=x.shape[1]
    w=np.linalg.solve(x.T@x+ridge*np.eye(d),x.T@y+ridge*np.eye(d))
    return {"matrix":w.tolist(),"objective":float(np.sum((x@w-y)**2)+ridge*np.sum((w-np.eye(d))**2))}

def transform(features, model):
    return np.asarray(features,dtype=float)@np.asarray(model["matrix"],dtype=float)
