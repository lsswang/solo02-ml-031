import json
from pathlib import Path
from adapter import fit
from recognizer import build,predict

def run(path):
    data=json.loads(Path(path).read_text())
    a=fit(data["fit_features"],data["fit_targets"],data["anchors"],data["ridge"])
    r=build(data["support"],data["calibration"],a,data["shrinkage"])
    return {"adapter":a,"predictions":predict(data["queries"],r,data["alpha"])}

if __name__=="__main__":
    import sys
    print(json.dumps(run(sys.argv[1]),ensure_ascii=False,indent=2))
